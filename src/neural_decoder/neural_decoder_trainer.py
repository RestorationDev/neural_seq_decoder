import os
import pickle
import sys
import time

from edit_distance import SequenceMatcher
import hydra
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .model import GRUDecoder
from .dataset import SpeechDataset


def apply_label_smoothing(logits, smoothing=0.0, num_classes=None):
    """
    Apply label smoothing to logits for CTC loss.
    
    Args:
        logits: Tensor of shape [batch, seq_len, num_classes] or [seq_len, batch, num_classes]
        smoothing: Label smoothing factor (0.0 = no smoothing, 1.0 = uniform distribution)
                   Must be in range [0.0, 1.0). Values >= 1.0 will be clamped to 0.99.
        num_classes: Number of classes (including blank token)
    
    Returns:
        Log probabilities (smoothed if smoothing > 0, otherwise standard log_softmax)
    """
    # Always convert to log probabilities first
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    
    if smoothing <= 0.0:
        return log_probs
    
    # Validate and clamp smoothing to valid range [0.0, 1.0)
    # If smoothing >= 1.0, torch.log(1.0 - smoothing) would be invalid (log of non-positive)
    if smoothing >= 1.0:
        import warnings
        warnings.warn(f"Label smoothing value {smoothing} >= 1.0 is invalid. Clamping to 0.99.")
        smoothing = 0.99
    
    if num_classes is None:
        num_classes = logits.shape[-1]
    
    # Create uniform distribution in log space
    uniform_log_prob = torch.log(torch.ones_like(log_probs) / num_classes)
    
    # Apply smoothing: (1 - smoothing) * log_probs + smoothing * uniform
    # In log space: log((1-smoothing) * exp(log_probs) + smoothing * exp(uniform))
    # Using logsumexp for numerical stability
    # Convert smoothing to tensor if it's a float
    if not isinstance(smoothing, torch.Tensor):
        smoothing = torch.tensor(smoothing, dtype=log_probs.dtype, device=log_probs.device)
    
    smoothed_log_probs = torch.logsumexp(
        torch.stack([
            torch.log(1.0 - smoothing) + log_probs,
            torch.log(smoothing) + uniform_log_prob
        ]),
        dim=0
    )
    
    return smoothed_log_probs


def getDatasetLoaders(
    datasetName,
    batchSize,
):
    print(f"[DEBUG] Loading dataset from: {datasetName}", flush=True)
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)
    print(f"[DEBUG] Dataset loaded. Train samples: {len(loadedData['train'])}, Test samples: {len(loadedData['test'])}", flush=True)

    def _padding(batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)

        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )

    train_ds = SpeechDataset(loadedData["train"], transform=None)
    test_ds = SpeechDataset(loadedData["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )

    print(f"[DEBUG] DataLoaders created. Train batches: {len(train_loader)}, Test batches: {len(test_loader)}", flush=True)
    return train_loader, test_loader, loadedData

def trainModel(args):
    print(f"[DEBUG] ===== Starting trainModel =====", flush=True)
    print(f"[DEBUG] Output directory: {args['outputDir']}", flush=True)
    print(f"[DEBUG] Dataset path: {args['datasetPath']}", flush=True)
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = "cuda"
    print(f"[DEBUG] Using device: {device}", flush=True)

    print(f"[DEBUG] Saving args to: {args['outputDir']}/args", flush=True)
    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)
    print(f"[DEBUG] Args saved", flush=True)

    print(f"[DEBUG] Loading dataset...", flush=True)
    trainLoader, testLoader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"],
    )
    print(f"[DEBUG] Dataset loaded successfully", flush=True)

    print(f"[DEBUG] Creating model...", flush=True)
    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=len(loadedData["train"]),
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
        use_layer_norm=args.get("use_layer_norm", False),
        layer_norm_position=args.get("layer_norm_position", "after_gru"),
    ).to(device)
    print(f"[DEBUG] Model created and moved to {device}", flush=True)

    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    label_smoothing = args.get("label_smoothing", 0.0)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args["lrStart"],
        betas=(0.9, 0.999),
        eps=0.1,
        weight_decay=args["l2_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=args["lrEnd"] / args["lrStart"],
        total_iters=args["nBatch"],
    )

    # --train--
    print(f"[DEBUG] Starting training loop. Total batches: {args['nBatch']}", flush=True)
    testLoss = []
    testCER = []
    startTime = time.time()
    for batch in range(args["nBatch"]):
        if batch == 0:
            print(f"[DEBUG] Starting batch 0...", flush=True)
        elif batch % 10 == 0:
            print(f"[DEBUG] Processing batch {batch}...", flush=True)
        
        model.train()

        X, y, X_len, y_len, dayIdx = next(iter(trainLoader))
        if batch == 0:
            print(f"[DEBUG] Batch 0: Got data. Shapes - X: {X.shape}, y: {y.shape}", flush=True)
        X, y, X_len, y_len, dayIdx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            dayIdx.to(device),
        )

        # Noise augmentation is faster on GPU
        if args["whiteNoiseSD"] > 0:
            X += torch.randn(X.shape, device=device) * args["whiteNoiseSD"]

        if args["constantOffsetSD"] > 0:
            X += (
                torch.randn([X.shape[0], 1, X.shape[2]], device=device)
                * args["constantOffsetSD"]
            )

        # Compute prediction error
        if batch == 0:
            print(f"[DEBUG] Batch 0: Running forward pass...", flush=True)
        pred = model.forward(X, dayIdx)
        if batch == 0:
            print(f"[DEBUG] Batch 0: Forward pass complete. Pred shape: {pred.shape}", flush=True)
        
        # Apply label smoothing if enabled
        if label_smoothing > 0.0:
            pred_log_probs = apply_label_smoothing(
                pred, 
                smoothing=label_smoothing, 
                num_classes=args["nClasses"] + 1  # +1 for CTC blank
            )
        else:
            pred_log_probs = pred.log_softmax(2)

        loss = loss_ctc(
            torch.permute(pred_log_probs, [1, 0, 2]),
            y,
            ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
            y_len,
        )
        loss = torch.sum(loss)

        # Backpropagation
        if batch == 0:
            print(f"[DEBUG] Batch 0: Loss computed: {loss.item():.6f}. Starting backprop...", flush=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if batch == 0:
            print(f"[DEBUG] Batch 0: Backprop complete. Batch 0 finished!", flush=True)

        # print(endTime - startTime)

        # Eval
        test_interval = args.get("testInterval", 100)
        skip_test_eval = args.get("skip_test_eval", False)
        if batch % test_interval == 0 and not skip_test_eval:
            with torch.no_grad():
                model.eval()
                allLoss = []
                total_edit_distance = 0
                total_seq_length = 0
                for X, y, X_len, y_len, testDayIdx in testLoader:
                    X, y, X_len, y_len, testDayIdx = (
                        X.to(device),
                        y.to(device),
                        X_len.to(device),
                        y_len.to(device),
                        testDayIdx.to(device),
                    )

                    pred = model.forward(X, testDayIdx)
                    
                    # Apply label smoothing if enabled (for consistency, but typically not needed in eval)
                    if label_smoothing > 0.0:
                        pred_log_probs = apply_label_smoothing(
                            pred,
                            smoothing=label_smoothing,
                            num_classes=args["nClasses"] + 1
                        )
                    else:
                        pred_log_probs = pred.log_softmax(2)
                    
                    loss = loss_ctc(
                        torch.permute(pred_log_probs, [1, 0, 2]),
                        y,
                        ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
                        y_len,
                    )
                    loss = torch.sum(loss)
                    # Convert to numpy safely - handle case where numpy might not be available
                    try:
                        allLoss.append(loss.cpu().detach().numpy())
                    except (RuntimeError, AttributeError) as e:
                        # Fallback: convert to Python float if numpy fails
                        allLoss.append(float(loss.cpu().detach().item()))

                    adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(
                        torch.int32
                    )
                    for iterIdx in range(pred.shape[0]):
                        decodedSeq = torch.argmax(
                            torch.tensor(pred[iterIdx, 0 : adjustedLens[iterIdx], :]),
                            dim=-1,
                        )  # [num_seq,]
                        decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                        # Convert to numpy safely
                        try:
                            decodedSeq = decodedSeq.cpu().detach().numpy()
                        except (RuntimeError, AttributeError):
                            # Fallback: convert to list then numpy array
                            decodedSeq = decodedSeq.cpu().detach().tolist()
                            decodedSeq = np.array(decodedSeq)
                        decodedSeq = np.array([i for i in decodedSeq if i != 0])

                        # Convert to numpy safely
                        try:
                            trueSeq = np.array(
                                y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
                            )
                        except (RuntimeError, AttributeError):
                            # Fallback: convert to list then numpy array
                            trueSeq = y[iterIdx][0 : y_len[iterIdx]].cpu().detach().tolist()
                            trueSeq = np.array(trueSeq)

                        matcher = SequenceMatcher(
                            a=trueSeq.tolist(), b=decodedSeq.tolist()
                        )
                        total_edit_distance += matcher.distance()
                        total_seq_length += len(trueSeq)

                avgDayLoss = np.sum(allLoss) / len(testLoader)
                cer = total_edit_distance / total_seq_length

                endTime = time.time()
                print(
                    f"batch {batch}, ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}, time/batch: {(endTime - startTime)/100:>7.3f}"
                )
                startTime = time.time()

            if len(testCER) > 0 and cer < np.min(testCER):
                torch.save(model.state_dict(), args["outputDir"] + "/modelWeights")
            testLoss.append(avgDayLoss)
            testCER.append(cer)

            tStats = {}
            tStats["testLoss"] = np.array(testLoss)
            tStats["testCER"] = np.array(testCER)

            with open(args["outputDir"] + "/trainingStats", "wb") as file:
                pickle.dump(tStats, file)


def loadModel(modelDir, nInputLayers=24, device="cuda"):
    modelWeightPath = modelDir + "/modelWeights"
    with open(modelDir + "/args", "rb") as handle:
        args = pickle.load(handle)

    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=nInputLayers,
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
        use_layer_norm=args.get("use_layer_norm", False),
        layer_norm_position=args.get("layer_norm_position", "after_gru"),
    ).to(device)

    model.load_state_dict(torch.load(modelWeightPath, map_location=device))
    return model


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg):
    print(f"[DEBUG] ===== main() called =====", flush=True)
    print(f"[DEBUG] Current working directory: {os.getcwd()}", flush=True)
    cfg.outputDir = os.getcwd()
    print(f"[DEBUG] Calling trainModel...", flush=True)
    trainModel(cfg)
    print(f"[DEBUG] ===== trainModel completed =====", flush=True)

if __name__ == "__main__":
    main()