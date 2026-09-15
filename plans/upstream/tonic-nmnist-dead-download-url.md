# N-MNIST download URL returns 404 (dataset cannot be fetched)

**Affects:** `tonic.datasets.NMNIST`
**Confirmed on:** tonic 1.4.3 and 1.6.0
**Severity:** dataset unobtainable through tonic

## Summary

`NMNIST`'s pinned download URLs point at Mendeley file endpoints that no longer
resolve:

```python
# tonic/datasets/nmnist.py
base_url = "https://data.mendeley.com/public-files/datasets/468j46mzdv/files/"
train_url = base_url + "39c25547-014b-4137-a934-9d29fa53c7a0/file_downloaded"
test_url  = base_url + "05a4d654-7e03-4c15-bdfa-9bb2bcbea494/file_downloaded"
```

Both return **HTTP 404**:

```console
$ curl -sSL -o /dev/null -w "%{http_code}\n" -r 0-1000 \
    "https://data.mendeley.com/public-files/datasets/468j46mzdv/files/39c25547-014b-4137-a934-9d29fa53c7a0/file_downloaded"
404
```

So `NMNIST(save_to=..., train=True)` cannot complete on a cold cache.

## Context

Mendeley Data appears to have changed its file-serving URL scheme; the dataset
record itself (`468j46mzdv`) is still the canonical N-MNIST location, so this
looks like a URL-scheme change rather than a withdrawn dataset. The current
`/public-files/datasets/<id>/files/<uuid>/file_downloaded` form no longer
resolves.

## Suggested fix

Re-derive the download links from the current Mendeley record and update
`train_url` / `test_url`. If Mendeley's per-file URLs are not stable enough to
pin, resolving them through the record's API at download time — or documenting a
mirror — would be more durable than hard-coded UUIDs.

## Related: the DVS figshare endpoints are also not serving

Not necessarily a tonic bug, but relevant to anyone hitting download failures.
`DVSGesture` and `CIFAR10DVS` fetch from figshare, which answers with **HTTP 202
and an empty body** rather than the file:

```console
$ curl -sI -r 0-1023 -o /dev/null -w "%{http_code}\n" \
    https://figshare.com/ndownloader/files/38022171
202
```

tonic then reports `RuntimeError: File not found or corrupted.` — accurate, but
it does not distinguish "the server declined to serve this" from "the file is
corrupt", which sends people looking for a cache problem they do not have.
Distinguishing a non-200 response from a checksum mismatch in the error message
would save time.

Between these, on a cold cache the only event datasets we could fetch at all
were SHD and SSC — and those hit the `float16` timestamp bug reported separately.
