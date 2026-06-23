"""异步图像增强缓冲区：将增强从 DataLoader worker 分离到独立线程池。"""
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import torch

from dataset import _apply_augmentations


def _augment_one(packed):
    """增强一条 packed [T, 3, 320, 960] uint8。"""
    T = packed.shape[0]
    out = torch.empty_like(packed)
    panels = [(0, 320), (320, 640), (640, 960)]
    for t in range(T):
        for w0, w1 in panels:
            panel = packed[t, :, :, w0:w1]               # [3, 320, 320] uint8
            panel_np = panel.permute(1, 2, 0).numpy()     # [H, W, 3]
            panel_np = _apply_augmentations(panel_np)
            out[t, :, :, w0:w1] = torch.from_numpy(panel_np).permute(2, 0, 1)
    return out


def _augment_batch(batch):
    """增强一个 batch。packed: [B, T, 3, 320, 960]"""
    pose, packed, labels, kf_labels = batch
    B = packed.shape[0]
    parts = [_augment_one(packed[b]) for b in range(B)]
    aug_packed = torch.stack(parts, dim=0)
    return pose, aug_packed, labels, kf_labels


class AugmentBuffer:
    """异步增强缓冲区。

    包装 DataLoader，用线程池在后台对已取出的 batch 做增强，
    维持一个预增强队列，避免 GPU 等待 CPU 增强。
    """
    def __init__(self, dataloader, num_threads=4, prefetch=3):
        self.dataloader = dataloader
        self.pool = ThreadPoolExecutor(max_workers=num_threads)
        self.prefetch = prefetch

    def __iter__(self):
        buf: Queue = Queue(maxsize=self.prefetch)
        raw_iter = iter(self.dataloader)

        def _submit():
            try:
                batch = next(raw_iter)
                buf.put(self.pool.submit(_augment_batch, batch))
            except StopIteration:
                buf.put(None)

        # 预填充缓冲区
        for _ in range(self.prefetch):
            _submit()

        # 消费循环
        while True:
            item = buf.get()
            if item is None:
                break
            yield item.result()
            _submit()

    def __len__(self):
        return len(self.dataloader)
