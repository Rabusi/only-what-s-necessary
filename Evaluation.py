import os
import glob
import numpy as np
import torch

from library.Dataloader import get_single_test
from Config import cfg
from models.model import get_model
from library import Helpers
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


# GT_RANGES = {
#     "Test001": [(61, 180)],
#     "Test002": [(95, 180)],
#     "Test003": [(1, 146)],
#     "Test004": [(31, 180)],
#     "Test005": [(1, 129)],
#     "Test006": [(1, 159)],
#     "Test007": [(46, 180)],
#     "Test008": [(1, 180)],
#     "Test009": [(1, 120)],
#     "Test010": [(1, 150)],
#     "Test011": [(1, 180)],
#     "Test012": [(88, 180)],
# }

GT_RANGES = {
    "Test001": [(60, 152)],
    "Test002": [(50, 175)],
    "Test003": [(91, 200)],
    "Test004": [(31, 168)],
    "Test005": [(5, 90), (140, 200)],
    "Test006": [(1, 100), (110, 200)],
    "Test007": [(1, 175)],
    "Test008": [(1, 94)],
    "Test009": [(1, 48)],
    "Test010": [(1, 140)],
    "Test011": [(70, 165)],
    "Test012": [(130, 200)],
    "Test013": [(1, 156)],
    "Test014": [(1, 200)],
    "Test015": [(138, 200)],
    "Test016": [(123, 200)],
    "Test017": [(1, 47)],
    "Test018": [(54, 120)],
    "Test019": [(64, 138)],
    "Test020": [(45, 175)],
    "Test021": [(31, 200)],
    "Test022": [(16, 107)],
    "Test023": [(8, 165)],
    "Test024": [(50, 171)],
    "Test025": [(40, 135)],
    "Test026": [(77, 144)],
    "Test027": [(10, 122)],
    "Test028": [(105, 200)],
    "Test029": [(1, 15), (45, 113)],
    "Test030": [(175, 200)],
    "Test031": [(1, 180)],
    "Test032": [(1, 52), (65, 115)],
    "Test033": [(5, 165)],
    "Test034": [(1, 121)],
    "Test035": [(86, 200)],
    "Test036": [(15, 108)],
}


def get_gt_labels(folder_name, num_frames):
    labels = np.zeros(num_frames, dtype=np.int32)

    if folder_name not in GT_RANGES:
        raise ValueError(f"No GT range found for {folder_name}")

    for start, end in GT_RANGES[folder_name]:
        labels[start - 1:end] = 1   # MATLAB 1-based inclusive -> Python slice

    return labels


def evaluate():
    print("Entered evaluate()", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Using device: {device}", flush=True)

    model = get_model(reload_model=True)
    model = model.to(device)
    model.eval()
    print(f"Model device: {next(model.parameters()).device}", flush=True)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    save_dir = os.path.join(cfg.logs, cfg.run_id, "evaluation_scores")
    reg_dir = os.path.join(save_dir, "Regularity score")
    reg_gt_dir = os.path.join(save_dir, "Regularity score gt")
    ano_dir = os.path.join(save_dir, "Anomalous score")
    ano_gt_dir = os.path.join(save_dir, "Anomalous score gt")
    roc_dir = os.path.join(save_dir, "ROC curves")
    npy_dir = os.path.join(save_dir, "npy")

    for d in [save_dir, reg_dir, reg_gt_dir, ano_dir, ano_gt_dir, roc_dir, npy_dir]:
        os.makedirs(d, exist_ok=True)

    test_folders = sorted([
        p for p in glob.glob(os.path.join(cfg.UCSD_test_path, "Test*"))
        if os.path.isdir(p) and not os.path.basename(p).endswith("_gt")
    ])
    print(f"Found {len(test_folders)} test folders", flush=True)

    all_y_true = []
    all_y_score = []
    auc_dict = {}

    seq_len = 10
    batch_size = 8

    for test_folder in test_folders:
        folder_name = os.path.basename(test_folder)
        print(f"\nEvaluating {folder_name}", flush=True)

        test_images = get_single_test(test_folder)   # expected shape: [N, H, W, C]
        num_frames = test_images.shape[0]
        print(f"{folder_name}: loaded {num_frames} frames", flush=True)

        if num_frames < seq_len:
            print(f"{folder_name}: skipped, less than {seq_len} frames", flush=True)
            continue

        num_sequences = num_frames - seq_len + 1

        # [N, H, W, C] -> [N, C, H, W]
        frames = torch.from_numpy(test_images).permute(0, 3, 1, 2).float().to(device, non_blocking=True)

        # Create sliding windows: [num_sequences, C, H, W, seq_len]
        windows = frames.unfold(0, seq_len, 1)

        # Rearrange to [num_sequences, seq_len, C, H, W]
        windows = windows.permute(0, 4, 1, 2, 3).contiguous()

        seq_recon_cost = []

        with torch.no_grad():
            for start_idx in range(0, num_sequences, batch_size):
                end_idx = min(start_idx + batch_size, num_sequences)
                batch = windows[start_idx:end_idx]

                if start_idx == 0:
                    print(f"{folder_name}: first batch shape = {batch.shape}", flush=True)
                    print(f"{folder_name}: first batch device = {batch.device}", flush=True)

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    recon_batch = model(batch)

                diff = batch - recon_batch
                cost = torch.norm(diff.reshape(diff.shape[0], -1), dim=1)
                seq_recon_cost.append(cost.detach().cpu())

        seq_recon_cost = torch.cat(seq_recon_cost).numpy()

        anomaly_score_seq = (
            (seq_recon_cost - seq_recon_cost.min()) /
            (seq_recon_cost.max() - seq_recon_cost.min() + 1e-8)
        )

        sa = anomaly_score_seq
        sr = 1.0 - anomaly_score_seq
        sr_smooth = Helpers.movingaverage(sr, 5)
        sa_smooth = Helpers.movingaverage(sa, 5)

        frame_scores = np.zeros(num_frames, dtype=np.float32)
        frame_scores[seq_len - 1:] = anomaly_score_seq

        gt_labels = get_gt_labels(folder_name, num_frames)

        y_true = gt_labels[seq_len - 1:]
        y_score = frame_scores[seq_len - 1:]

        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            print(f"{folder_name}: ROC AUC undefined because only one class is present", flush=True)
            auc = None
            fpr, tpr = None, None
        else:
            auc = roc_auc_score(y_true, y_score)
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_dict[folder_name] = auc
            all_y_true.extend(y_true.tolist())
            all_y_score.extend(y_score.tolist())
            print(f"{folder_name}: AUC ROC = {auc:.4f}", flush=True)

        # Regularity plot
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(len(sr_smooth)), sr_smooth, label="Regularity Score")
        plt.ylabel("Regularity Score")
        plt.xlabel("Sequence Index")
        plt.title(f"{folder_name} | AUC={auc:.4f}" if auc is not None else f"{folder_name} | AUC undefined")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(reg_dir, f"{folder_name}_regularity_score.png"))
        plt.close()

        # Anomalous plot
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(len(sa_smooth)), sa_smooth, label="Anomalous Score")
        plt.ylabel("Anomalous Score")
        plt.xlabel("Sequence Index")
        plt.title(f"{folder_name} | AUC={auc:.4f}" if auc is not None else f"{folder_name} | AUC undefined")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(ano_dir, f"{folder_name}_anomalous_score.png"))
        plt.close()

        # Regularity + GT
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(seq_len - 1, num_frames), sr, label="Regularity Score")
        plt.plot(np.arange(num_frames), gt_labels, label="Ground Truth", alpha=0.7)
        plt.ylabel("Score / Label")
        plt.xlabel("Frame Index")
        plt.title(f"{folder_name} Regularity vs GT")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(reg_gt_dir, f"{folder_name}_regularity_with_gt.png"))
        plt.close()

        # Anomalous + GT
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(seq_len - 1, num_frames), sa, label="Anomalous Score")
        plt.plot(np.arange(num_frames), gt_labels, label="Ground Truth", alpha=0.7)
        plt.ylabel("Score / Label")
        plt.xlabel("Frame Index")
        plt.title(f"{folder_name} Anomalous vs GT")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(ano_gt_dir, f"{folder_name}_anomalous_with_gt.png"))
        plt.close()

        # ROC per folder
        if auc is not None:
            plt.figure(figsize=(5, 5))
            plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
            plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"{folder_name} ROC Curve")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(roc_dir, f"{folder_name}_roc_curve.png"))
            plt.close()

        # Save arrays
        np.save(os.path.join(npy_dir, f"{folder_name}_seq_recon_cost.npy"), seq_recon_cost)
        np.save(os.path.join(npy_dir, f"{folder_name}_regularity_score.npy"), sr)
        np.save(os.path.join(npy_dir, f"{folder_name}_regularity_score_smooth.npy"), sr_smooth)
        np.save(os.path.join(npy_dir, f"{folder_name}_frame_anomaly_score.npy"), frame_scores)
        np.save(os.path.join(npy_dir, f"{folder_name}_gt_labels.npy"), gt_labels)

        print(f"{folder_name}: done", flush=True)

        del frames, windows
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Overall ROC
    if len(np.unique(all_y_true)) >= 2:
        all_y_true = np.array(all_y_true)
        all_y_score = np.array(all_y_score)

        overall_auc = roc_auc_score(all_y_true, all_y_score)
        overall_fpr, overall_tpr, _ = roc_curve(all_y_true, all_y_score)

        print(f"\nOverall AUC ROC = {overall_auc:.4f}", flush=True)

        plt.figure(figsize=(5, 5))
        plt.plot(overall_fpr, overall_tpr, label=f"AUC = {overall_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Overall ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(roc_dir, "overall_roc_curve.png"))
        plt.close()
    else:
        overall_auc = None
        print("\nOverall AUC ROC undefined because only one class is present overall", flush=True)

    with open(os.path.join(save_dir, "auc_results.txt"), "w") as f:
        for folder_name in sorted(auc_dict.keys()):
            f.write(f"{folder_name}: {auc_dict[folder_name]:.6f}\n")
        if overall_auc is not None:
            f.write(f"Overall AUC ROC: {overall_auc:.6f}\n")
        else:
            f.write("Overall AUC ROC: undefined\n")

    if len(auc_dict) > 0:
        folder_names = list(auc_dict.keys())
        auc_values = [auc_dict[name] for name in folder_names]

        plt.figure(figsize=(10, 5))
        plt.bar(folder_names, auc_values)
        plt.xticks(rotation=45)
        plt.ylabel("AUC ROC")
        plt.xlabel("Test Folder")
        plt.title("AUC ROC per Test Folder")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "auc_barplot.png"))
        plt.close()

    print("\nAll test folders evaluated", flush=True)
    print(f"Results saved in: {save_dir}", flush=True)


if __name__ == "__main__":
    print("Evaluation started", flush=True)
    evaluate()