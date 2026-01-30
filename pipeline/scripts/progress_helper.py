"""Progress helper utilities."""

try:
    from tqdm import tqdm as tqdm
except Exception:  # pragma: no cover - optional dependency
    def tqdm(iterable, **kwargs):
        return iterable
