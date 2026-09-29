#!/usr/bin/env python3
"""Fetch the third-party data and code that this repository reads but does not ship.

The repository contains no data files and none of the KSPC authors' code.  This script fetches each file from
its original public source into the path the code expects (under materials/) and checks it against the
SHA-256 of the file used for the paper (MD5 for Shapes3D: the code checks that file by MD5 and no SHA-256 was
recorded).  A file already in place with the right checksum is left alone, so the script can be rerun.

    python scripts/download_data.py                     # every dataset except the optional Shapes3D file
    python scripts/download_data.py --list              # what is fetched, from where, under which terms
    python scripts/download_data.py --only twins ihdp   # only the named datasets
    python scripts/download_data.py --only shapes3d     # the 268 MB Shapes3D file, needed only for SCM-4
    python scripts/download_data.py --check-urls        # HEAD every source and compare sizes; fetches nothing
    python scripts/download_data.py --from-local DIR    # copy from a tree laid out like materials/, then verify
    python scripts/download_data.py --root DIR          # write under DIR instead of this repository

Exit status: 0 when every requested file is in place and verified; 1 when any checksum differs; 3 when a
file could not be obtained (source unreachable, file absent from --from-local, git not installed; the
script then prints the manual step); 2 on a usage error.

Standard library only, Python 3.8 or later.  The KSPC code is fetched with git.  If Python has no CA
certificates (the python.org macOS build), a system CA bundle is used, or certifi when it is installed.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import os
import shutil
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "download_data.py (Proximal Balancing reproduction; Python urllib)"
CHUNK = 1 << 20


# ----------------------------------------------------------------------------------------------- the table


@dataclass(frozen=True)
class File:
    dest: str                     # relative to the repository root
    size: int                     # bytes
    sha256: Optional[str]
    urls: Tuple[str, ...]         # tried in order
    md5: Optional[str] = None     # checked only when no SHA-256 is recorded


@dataclass(frozen=True)
class GitTree:
    url: str
    commit: str
    dest: str                     # directory, relative to the repository root
    copy: Tuple[str, ...]         # top-level entries copied out of the checkout
    digest_over: Tuple[str, ...]  # entries whose files enter the tree digest
    tree_sha256: str              # sha256 of the lines "<sha256 of file>  <path>\n", sorted by path
    note: Tuple[str, str, str] = ("", "", "")   # (file name, exact text, sha256) written beside the code


@dataclass(frozen=True)
class Dataset:
    name: str
    title: str
    source: str
    terms: str
    cite: str
    files: Tuple[File, ...] = ()
    tree: Optional[GitTree] = None
    default: bool = True
    manual: str = ""              # printed when a file cannot be obtained automatically


def gh(repo: Tuple[str, str], path: str) -> str:
    """A file of a GitHub repository at a fixed commit."""
    return f"https://raw.githubusercontent.com/{repo[0]}/{repo[1]}/{path}"


RWD = "materials/real_world_data"
CEVAE = ("AMLab-Amsterdam/CEVAE", "9081f863e24ce21bd34c8d6a41bf0edc7d1b65dd")
CAUSALLIB = ("BiomedSciAI/causallib", "267e0c72df81ea8264ece46496107f99a599b60d")
LALONDE = ("xuyiqing/lalonde", "c179a80e397178185983c272e1c1105d7ca53bf3")
SPC = ("qkrcks0218/SingleProxyControl", "37270e8ef8782d41d87c60f94ace6384e908615d")
DEHEJIA = "https://users.nber.org/~rdehejia/data/"


def _twins(name: str, size: int, sha: str) -> File:
    return File(f"{RWD}/twins/{name}", size, sha, (gh(CEVAE, f"datasets/TWINS/{name}"),))


def _acic(name: str, size: int, sha: str) -> File:
    return File(f"{RWD}/acic2016/{name}", size, sha,
                (gh(CAUSALLIB, f"causallib/datasets/data/acic_challenge_2016/{name}"),))


def _lalonde(name: str, upstream: str, size: int, sha: str, dehejia: bool = False) -> File:
    return File(f"{RWD}/lalonde/{name}", size, sha,
                (gh(LALONDE, upstream),) + ((DEHEJIA + name,) if dehejia else ()))


# The provenance note that sat beside the KSPC code when the paper's KSPC runs were made, byte for byte:
# results/kspc_design/FROZEN_v1.json records its sha256, so it is restored exactly rather than rewritten.
KSPC_PROVENANCE = (
    "# Kernel Single Proxy Control, authors' code\n\n"
    "- Source: https://github.com/liyuan9988/KernelSingleProxy\n"
    "- Commit: a5283d6210799db2d04b6e288442ede0aad72305 (cloned 2026-09-22 21:36 CDT, depth 1)\n"
    "- Paper: Xu and Gretton, Kernel Single Proxy Control for Deterministic Confounding, arXiv 2308.04585 v4.\n"
    "- License: none in the repository at this commit. Used unchanged, only as a comparator inside this private\n"
    "  research workspace; not redistributed.\n"
    "- Copied: `src/`, `configs/`, `main.py`, `README.md`, `requirements.txt`. Nothing edited.\n"
    "tree sha256 (first 16): 0c844171912d50c7\n")


DATASETS: Tuple[Dataset, ...] = (
    Dataset(
        "twins", "Twins: US same-sex twin births 1989-1991, as packaged by the CEVAE authors",
        f"https://github.com/{CEVAE[0]}/tree/{CEVAE[1]}/datasets/TWINS",
        "no licence file in the repository; derived from the NBER public-use linked birth/infant death files",
        "Louizos et al. (2017), NeurIPS; Almond, Chay and Lee (2005), QJE 120(3):1031-1083",
        files=(
            _twins("twin_pairs_X_3years_samesex.csv", 13812350,
                   "6ec6263e678f67205d584ebda083173c1ec44fb46ea7cfaafe3c87a31f612b86"),
            _twins("twin_pairs_T_3years_samesex.csv", 1409691,
                   "fbe6eb1cad27794c61cbbd8b11b72a26e96549e59d64fb88b07b30aaf2bae07d"),
            _twins("twin_pairs_Y_3years_samesex.csv", 987735,
                   "d03428d23ed308673f9151e3934aa7712df27c1fed244ed44ecc80a98a60836b"),
            _twins("covar_desc.txt", 2113, "51d126a3a16a7f46bfe42536bd0ae1e2e31dc840b0e5000b595b76b7aee672c0"),
            _twins("covar_type.txt", 927, "fe8a7d72b53c92f74210b52743e047d150d1d38a475dbe142887ea5de444788a"),
        )),
    Dataset(
        "ihdp", "IHDP: the 100 semi-synthetic replications of Hill (2011), npz files of Johansson et al.",
        "https://www.fredjo.com/",
        "no licence or terms stated on the distribution page",
        "Hill (2011), JCGS 20(1):217-240; Johansson, Shalit and Sontag (2016), ICML",
        files=(
            File(f"{RWD}/ihdp/ihdp_npci_1-100.train.npz", 16129570,
                 "750697c71b4f8d7a3aafff771b56a4ac4cd83ec649bf69afb04f8a5aee41a240",
                 ("https://www.fredjo.com/files/ihdp_npci_1-100.train.npz",)),
            File(f"{RWD}/ihdp/ihdp_npci_1-100.test.npz", 1801570,
                 "a70a8acbcc4e8deb677cc9bf9e9dabeb17caaa37cdbb1d7ba06be7ffb929c41c",
                 ("https://www.fredjo.com/files/ihdp_npci_1-100.test.npz",)),
        )),
    Dataset(
        "rhc", "RHC: right heart catheterization in the SUPPORT study",
        "https://hbiostat.org/data/repo/rhc.html",
        "hbiostat.org permits anyone to use its data sets; cite the original paper and credit "
        "hbiostat.org/data, courtesy of the Vanderbilt University Department of Biostatistics",
        "Connors et al. (1996), JAMA 276(11):889-897",
        files=(
            File(f"{RWD}/rhc/rhc.csv", 2320412,
                 "9ef4ab578be4b40ad5d97d3a7e08ffdc1f9f76aeeefee51b4996e4221556f8e8",
                 ("https://hbiostat.org/data/repo/rhc.csv",)),
        ),
        manual="hbiostat.org replaces files in place.  A checksum difference means the file changed after "
               "2026-08-23, when the copy used for the paper was taken; please open an issue rather than "
               "running on a different file."),
    Dataset(
        "acic2016", "ACIC 2016: covariates and 10 simulated realizations (the sample distributed with causallib)",
        f"https://github.com/{CAUSALLIB[0]}/tree/{CAUSALLIB[1]}/causallib/datasets/data/acic_challenge_2016",
        "Community Data License Agreement - Sharing 1.0 (the LICENSE file of that folder)",
        "Dorie et al. (2019), Statistical Science 34(1):43-68",
        files=(
            _acic("x.csv", 697786, "0d6387ad45d23e54b11cbf967248fb68ad5e578db80d193208b8b4c98b1ed0c1"),
            _acic("zymu_1.csv", 339560, "002fad96bb55d54edec08a0af46ef4f8ad69e0cfe6294ab48f351f22976a6e76"),
            _acic("zymu_2.csv", 334091, "bc3a03bb461bb1629add09c01dfc77053cf36e8faf8e48a7645b35ddb4005d1c"),
            _acic("zymu_3.csv", 343144, "8090562d3fcf5a3b02ff5493592a2880b95b106e1c12e2c3a6e6c28ae8468ff6"),
            _acic("zymu_4.csv", 341908, "61b405c545a451b7e2ce31614150834f94aa4b27069e2755876ed6e11cfee5fd"),
            _acic("zymu_5.csv", 348813, "e350ac76d0b8e83f45f7f04a4b76b9a32a0d53098452075617e61ce62ad0463a"),
            _acic("zymu_6.csv", 344205, "a059950f0b51056660364eca14524a3d3254f93c608d8a1df9240c2c4179a7b5"),
            _acic("zymu_7.csv", 347066, "ac102c3f34680ff901c30fb95985297eea0f20efaf02108f287ceec60b6a2a1f"),
            _acic("zymu_8.csv", 336120, "bbcde6ad666640adcd39fd50cab00865d22b5c463ce9db6d0940d8eccd5afcf3"),
            _acic("zymu_9.csv", 341289, "02663b776fa87cd3d91ee7871e98c1b4d63579957b12ef9bb1ecf367b7afc366"),
            _acic("zymu_10.csv", 353155, "c8ee8ca537ef4315a0ee5753bd01d792a1b3cdacb1fd69e1288b2bb89944d1b7"),
        )),
    Dataset(
        "lalonde", "LaLonde: NSW men with CPS/PSID comparison groups, and the NSW AFDC women",
        f"https://github.com/{LALONDE[0]}/tree/{LALONDE[1]}/data",
        "MIT (the LICENSE of the Imbens-Xu repository, fetched with the data)",
        "LaLonde (1986), AER 76(4); Dehejia and Wahba (1999), JASA 94(448); Calonico and Smith (2017), "
        "JOLE 35(S1); Imbens and Xu (2024), arXiv:2406.00827",
        files=(
            _lalonde("NSW_AFDC_CS.dta", "data/cs/NSW_AFDC_CS.dta", 241635,
                     "f906f0f0e320085d5d89b0663b4a573f38085bd28358ef61b03c1aed7854cef9"),
            _lalonde("nsw.dta", "data/lalonde/nsw.dta", 22674,
                     "3ee014cb0010e2e7fb990ac497f4217968733e2c9abfa4de84e83f524898c24e", dehejia=True),
            _lalonde("nsw_dw.dta", "data/lalonde/nsw_dw.dta", 28598,
                     "d1bd2680a1c6f799f1c6d2455bf29633fdf19be01cb19490621c20a560b4e072", dehejia=True),
            _lalonde("cps_controls.dta", "data/lalonde/cps_controls.dta", 705546,
                     "80f0123eaf723870bd060ba9f8569617a7cb519a462dc20badbfb3443cd8374b", dehejia=True),
            _lalonde("cps_controls2.dta", "data/lalonde/cps_controls2.dta", 106134,
                     "7f1090da18d27d308ffaef9ba509ed040441c9bfd84ccf693c4d97f5c078abeb", dehejia=True),
            _lalonde("cps_controls3.dta", "data/lalonde/cps_controls3.dta", 20774,
                     "74fda0885a8ea7872adc41609d9edbf54d5cba3d5c494412e6ad6186afbc7352", dehejia=True),
            _lalonde("psid_controls.dta", "data/lalonde/psid_controls.dta", 111458,
                     "7beebae8928035d6abf662b994ce77a4fbea7ae6575037ff94e92fe099b43e14", dehejia=True),
            _lalonde("psid_controls2.dta", "data/lalonde/psid_controls2.dta", 13283,
                     "799a1178376177359c4e6421c973201fbe03590da71f5fe6ed2abfee53427a3e", dehejia=True),
            _lalonde("psid_controls3.dta", "data/lalonde/psid_controls3.dta", 7658,
                     "20a2b0ce7237c36135530e09d9bf9b2470fe816d00d49d11d659ec5350ffd322", dehejia=True),
            _lalonde("datasets.csv", "data/lalonde/datasets.csv", 789,
                     "6d0a97087d0f8824339a808332ab8645c9fb109fe30c7753f69ea377dee219be"),
            _lalonde("source.txt", "data/lalonde/source.txt", 329,
                     "68bfa976168d0bb3b1606a6a6d4fbd55acd32284f948ac3db7d8e5ea8abe0a9c"),
            _lalonde("dataframes.csv", "data/dataframes.csv", 1957,
                     "466502d3c21cce39ffe558354ae37d0a7e3af9b76b7dfca2319f1af7ed4149fe"),
            _lalonde("LICENSE", "LICENSE", 1083, "5a329e855bdb819c48132b127000d9736d4831e3e9b380f761c8ded647679fb7"),
            _lalonde("ReadMe.md", "ReadMe.md", 5589, "9f3ca82dc928a0487ad31b80ed69367f55494569cff6f020792992c050226b9a"),
        ),
        manual="The Dehejia-Wahba files are also on https://users.nber.org/~rdehejia/data/.nswdata2.html (tried "
               "automatically).  NSW_AFDC_CS.dta is also built by the replication package of Calonico and Smith "
               "(2017), doi:10.1086/692397."),
    Dataset(
        "zika", "Zika: birth rates of 673 Brazilian municipalities, 2013-2016 (the authors' replication file)",
        f"https://github.com/{SPC[0]}/tree/{SPC[1]}",
        "no licence file in the authors' replication repository; built from Harvard Dataverse "
        "doi:10.7910/DVN/ENG0IY (CC0) and the 2010 IBGE census",
        "Park, Richardson and Tchetgen Tchetgen (2024), Biometrics 80(2):ujae027; Taddeo, Amorim and Aquino "
        "(2022), Statistics and Its Interface 15(4)",
        files=(
            File(f"{RWD}/zika/Zika_Brazil_2013.csv", 120670,
                 "a1a86446a6038ab79f7f29f767f54ae4e113288d9950c5ce75eeea59d7328664",
                 (gh(SPC, "Zika_Brazil_2013.csv"),)),
        ),
        manual="Only the authors' repository serves this exact file."),
    Dataset(
        "mnist", "MNIST in the Keras npz packaging",
        "https://keras.io/api/datasets/mnist/",
        "Creative Commons Attribution-Share Alike 3.0 (as stated in the Keras documentation)",
        "LeCun, Cortes and Burges, The MNIST database of handwritten digits",
        files=(
            File("materials/benchmarks/mnist/mnist.npz", 11490434,
                 "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1",
                 ("https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
                  "https://s3.amazonaws.com/img-datasets/mnist.npz")),
        )),
    Dataset(
        "shapes3d", "Shapes3D (3dshapes.h5): needed only to rebuild the SCM-4 image data sets",
        "https://github.com/google-deepmind/3d-shapes",
        "Apache 2.0 (the licence of the google-deepmind/3d-shapes repository, which distributes the file)",
        "Burgess and Kim (2018), 3D Shapes Dataset, https://github.com/deepmind/3dshapes-dataset/",
        files=(
            File("materials/benchmarks/shapes3d/3dshapes.h5", 267573662, None,
                 ("https://storage.googleapis.com/3d-shapes/3dshapes.h5",),
                 md5="099a2078d58cec4daad0702c55d06868"),
        ),
        default=False,
        manual="Fetched only when named (--only shapes3d): 268 MB, and code/family_v2_images.py then writes a "
               "5.9 GB image cache, so about 12 GB of free disk is needed."),
    Dataset(
        "kspc", "Kernel single proxy control: the authors' code, used unchanged as a comparator",
        "https://github.com/liyuan9988/KernelSingleProxy",
        "no licence in the repository at this commit, so it is fetched here and not redistributed",
        "Xu and Gretton, Kernel Single Proxy Control for Deterministic Confounding, AISTATS 2025, "
        "arXiv:2308.04585",
        tree=GitTree("https://github.com/liyuan9988/KernelSingleProxy.git",
                     "a5283d6210799db2d04b6e288442ede0aad72305",
                     "materials/external_code/KernelSingleProxy",
                     ("src", "configs", "main.py", "README.md", "requirements.txt"),
                     ("src", "configs", "main.py"),
                     "0c844171912d50c7f5bcabf4d9193bdbf734d1145889d1041e23194e3163644b",
                     ("PROVENANCE.md", KSPC_PROVENANCE,
                      "1d4dba38072c379333b7cdce9b8249ca5a1e1a4db036023962f561f268d2578e")),
        manual="By hand, with git: git clone https://github.com/liyuan9988/KernelSingleProxy, then git -C "
               "KernelSingleProxy checkout a5283d6210799db2d04b6e288442ede0aad72305, then copy src/, configs/, "
               "main.py, README.md and requirements.txt into materials/external_code/KernelSingleProxy/."),
)
NAMES = tuple(d.name for d in DATASETS)


# ----------------------------------------------------------------------------------------------- checksums


def _md5():
    try:
        return hashlib.md5(usedforsecurity=False)  # Python 3.9+; also works on FIPS builds
    except TypeError:
        return hashlib.md5()


class Digest:
    def __init__(self) -> None:
        self._sha256, self._md5, self.size = hashlib.sha256(), _md5(), 0

    def update(self, chunk: bytes) -> None:
        self._sha256.update(chunk)
        self._md5.update(chunk)
        self.size += len(chunk)

    @property
    def sha256(self) -> str:
        return self._sha256.hexdigest()

    @property
    def md5(self) -> str:
        return self._md5.hexdigest()


def digest_of(path: Path) -> Digest:
    d = Digest()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            d.update(chunk)
    return d


def matches(f: File, d: Digest) -> bool:
    if d.size != f.size:
        return False
    return d.sha256 == f.sha256 if f.sha256 else d.md5 == f.md5


def label(f: File, d: Optional[Digest]) -> str:
    if d is None:
        return ""
    return f"sha256:{d.sha256[:16]}" if f.sha256 else f"md5:{d.md5[:16]}"


def expected(f: File) -> str:
    return f"sha256 {f.sha256} ({f.size} bytes)" if f.sha256 else f"md5 {f.md5} ({f.size} bytes)"


def found(f: File, d: Digest) -> str:
    return f"sha256 {d.sha256} ({d.size} bytes)" if f.sha256 else f"md5 {d.md5} ({d.size} bytes)"


def _ignored(rel: Path) -> bool:
    return "__pycache__" in rel.parts or rel.name == ".DS_Store" or rel.suffix == ".pyc"


def tree_digest(base: Path, over: Tuple[str, ...]) -> Optional[str]:
    """sha256 of the lines "<sha256 of file>  <path>\\n" over every file under `over`, sorted by path."""
    rows = []
    for entry in over:
        p = base / entry
        if p.is_file():
            paths = [p]
        elif p.is_dir():
            paths = [q for q in p.rglob("*") if q.is_file() and not _ignored(q.relative_to(base))]
        else:
            return None
        rows += [(q.relative_to(base).as_posix(), digest_of(q).sha256) for q in paths]
    text = "".join(f"{h}  {p}\n" for p, h in sorted(rows))
    return hashlib.sha256(text.encode()).hexdigest()


# ----------------------------------------------------------------------------------------------- fetching


@dataclass
class Result:
    status: str                   # present | fetched | copied | mismatch | missing
    shown: str = ""
    problems: List[str] = field(default_factory=list)


def human(n: int) -> str:
    return f"{n / 1e6:.1f} MB" if n >= 100_000 else f"{n / 1e3:.1f} kB"


_CONTEXT: Optional[ssl.SSLContext] = None
CA_BUNDLES = ("/etc/ssl/cert.pem", "/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt")
CERT_HINT = ("Python could not verify HTTPS certificates.  With the python.org installer on macOS, run "
             "'Install Certificates.command' in the Python folder under /Applications; elsewhere set SSL_CERT_FILE "
             "to a CA bundle.  Then rerun.")


def ssl_context() -> ssl.SSLContext:
    """The default context, plus a system CA bundle when Python ships without one (the python.org macOS build
    before 'Install Certificates.command' has run), plus certifi if that is the only bundle available."""
    global _CONTEXT
    if _CONTEXT is None:
        ctx = ssl.create_default_context()
        if ctx.cert_store_stats()["x509_ca"] == 0 and not os.environ.get("SSL_CERT_FILE"):
            bundle = next((b for b in CA_BUNDLES if os.path.isfile(b)), None)
            if bundle is None:
                try:
                    import certifi  # optional; not required
                    bundle = certifi.where()
                except ImportError:
                    pass
            if bundle:
                ctx.load_verify_locations(cafile=bundle)
        _CONTEXT = ctx
    return _CONTEXT


def _reason(e: BaseException) -> str:
    text = str(getattr(e, "reason", e))
    return f"{text}\n           {CERT_HINT}" if "CERTIFICATE_VERIFY_FAILED" in text else text


def _open(src):
    if isinstance(src, Path):
        return open(src, "rb")
    req = urllib.request.Request(src, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(req, timeout=60, context=ssl_context())


def _stream(src, part: Path, size: int) -> Digest:
    d = Digest()
    progress = size >= 50 * CHUNK and sys.stderr.isatty()
    with _open(src) as inp, open(part, "wb") as out:
        for chunk in iter(lambda: inp.read(CHUNK), b""):
            out.write(chunk)
            d.update(chunk)
            if progress:
                print(f"\r    {min(100, 100 * d.size // size):3d}% of {human(size)}", end="", file=sys.stderr)
    if progress:
        print("\r" + " " * 30 + "\r", end="", file=sys.stderr)
    return d


def _unlink(p: Path) -> None:
    try:
        p.unlink()
    except FileNotFoundError:
        pass


def local_source(local: Path, dest: str) -> Path:
    return local / Path(dest).relative_to("materials")


def obtain_file(f: File, root: Path, local: Optional[Path]) -> Result:
    dest = root / f.dest
    stale = None
    if dest.is_file():
        d = digest_of(dest)
        if matches(f, d):
            return Result("present", label(f, d))
        stale = d
    sources = [local_source(local, f.dest)] if local else list(f.urls)
    part = dest.with_name(dest.name + ".part")
    problems, mismatch = [], False
    for src in sources:
        if isinstance(src, Path) and not src.is_file():
            problems.append(f"not found: {src}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        d = None
        for attempt in (1, 2):
            try:
                d = _stream(src, part, f.size)
                break
            except urllib.error.HTTPError as e:
                problems.append(f"{src}: HTTP {e.code} {e.reason}")
                break
            except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
                if attempt == 2 or "CERTIFICATE_VERIFY_FAILED" in str(e):
                    problems.append(f"{src}: {_reason(e)}")
                    break
                time.sleep(3)
        if d is None:
            _unlink(part)
            continue
        if matches(f, d):
            os.replace(part, dest)
            note = [f"replaced a file whose checksum differed ({label(f, stale)})"] if stale else []
            return Result("copied" if local else "fetched", label(f, d), note)
        _unlink(part)
        mismatch = True
        problems.append(f"{src}: checksum differs: got {found(f, d)}, expected {expected(f)}")
    if stale is not None:
        problems.insert(0, f"the file in place differs: {found(f, stale)}, expected {expected(f)}; "
                           f"it was left as it is")
        mismatch = True
    return Result("mismatch" if mismatch else "missing", "", problems)


def _git(*args: str, cwd: Path) -> str:
    cmd = ["git", "-c", "core.autocrlf=false", "-c", "advice.detachedHead=false", *args]
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout


def _checkout(url: str, commit: str, where: Path) -> None:
    where.mkdir()
    _git("init", "-q", cwd=where)
    _git("remote", "add", "origin", url, cwd=where)
    try:
        _git("fetch", "-q", "--depth", "1", "origin", commit, cwd=where)
    except subprocess.CalledProcessError:
        _git("fetch", "-q", "origin", cwd=where)     # a server that will not serve a commit by its id
    _git("checkout", "-q", "--detach", commit, cwd=where)
    head = _git("rev-parse", "HEAD", cwd=where).strip()
    if head != commit:
        raise RuntimeError(f"checked out {head}, expected {commit}")


def _write_note(t: GitTree, directory: Path) -> List[str]:
    """Write the note file beside the code, byte for byte; report it only when it had to be (re)written."""
    name, text, sha = t.note
    if not name:
        return []
    data = text.encode("utf-8")
    if hashlib.sha256(data).hexdigest() != sha:
        raise SystemExit(f"internal error: the embedded {name} does not match its recorded sha256")
    path = directory / name
    if path.is_file() and path.read_bytes() == data:
        return []
    path.write_bytes(data)
    return [f"wrote {name} (sha256:{sha[:16]})"]


def _only_note(t: GitTree, dest: Path) -> bool:
    """True when `dest` holds the note file, byte for byte as recorded, and nothing else (a .DS_Store aside): the
    state of a fresh clone of this repository, which ships the note but not the code.  Such a folder counts as
    absent; the code is added beside the note and no copy of the folder is kept."""
    name, _, sha = t.note
    if not name or not dest.is_dir() or dest.is_symlink():
        return False
    entries = [p for p in dest.iterdir() if p.name != ".DS_Store"]
    return (len(entries) == 1 and entries[0].name == name and entries[0].is_file() and not entries[0].is_symlink()
            and hashlib.sha256(entries[0].read_bytes()).hexdigest() == sha)


def obtain_tree(t: GitTree, root: Path, local: Optional[Path]) -> Result:
    dest = root / t.dest
    if dest.is_dir() and tree_digest(dest, t.digest_over) == t.tree_sha256:
        return Result("present", f"tree:{t.tree_sha256[:16]}", _write_note(t, dest))
    note_only = _only_note(t, dest)
    if local and not local_source(local, t.dest).is_dir():
        return Result("missing", problems=[f"not found: {local_source(local, t.dest)}"])
    if not local and shutil.which("git") is None:
        return Result("missing", problems=["git is not installed"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".fetch-", dir=dest.parent))
    try:
        if local:
            src = local_source(local, t.dest)
        else:
            src = staging / "checkout"
            try:
                _checkout(t.url, t.commit, src)
            except (subprocess.CalledProcessError, RuntimeError, OSError) as e:
                return Result("missing", problems=[f"git: {(getattr(e, 'stderr', '') or str(e)).strip()}"])
        tree = staging / "tree"
        tree.mkdir()
        for entry in t.copy:
            p = src / entry
            if p.is_dir():
                shutil.copytree(p, tree / entry, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
            elif p.is_file():
                shutil.copy2(p, tree / entry)
            else:
                return Result("missing", problems=[f"{entry} is not in {src}"])
        got = tree_digest(tree, t.digest_over)
        if got != t.tree_sha256:
            return Result("mismatch", problems=[f"tree sha256 {got}, expected {t.tree_sha256}"])
        status = "copied" if local else "fetched"
        if note_only:
            # a fresh clone: move the verified entries in beside the shipped note, which stays as it is
            for entry in t.copy:
                (tree / entry).rename(dest / entry)
            got = tree_digest(dest, t.digest_over)
            if got != t.tree_sha256:
                return Result("mismatch", problems=[f"tree sha256 {got} after moving into {dest}, "
                                                    f"expected {t.tree_sha256}"])
            return Result(status, f"tree:{got[:16]}")
        _write_note(t, tree)
        notes = []
        if dest.exists():
            kept = dest.with_name(dest.name + time.strftime(".replaced-%Y%m%d-%H%M%S"))
            dest.rename(kept)
            notes.append(f"the previous directory did not verify and was kept as {kept.name}")
        tree.rename(dest)
        return Result("copied" if local else "fetched", f"tree:{got[:16]}", notes)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


# ----------------------------------------------------------------------------------------------- commands


def list_datasets(selected: List[Dataset]) -> None:
    for ds in selected:
        what = (f"{len(ds.files)} file{'s' if len(ds.files) > 1 else ''}, {human(sum(f.size for f in ds.files))}"
                if ds.files
                else f"git tree at {ds.tree.commit[:12]}")
        print(f"{ds.name} ({what}; {'fetched by default' if ds.default else 'optional, fetched only when named'})")
        print(f"    {ds.title}")
        print(f"    source: {ds.source}")
        print(f"    terms:  {ds.terms}")
        print(f"    cite:   {ds.cite}")
        for f in ds.files:
            print(f"    -> {f.dest}  ({human(f.size)})")
        if ds.tree:
            extra = f", plus {ds.tree.note[0]}" if ds.tree.note[0] else ""
            print(f"    -> {ds.tree.dest}/  ({', '.join(ds.tree.copy)}{extra})")
        print()


def _head(url: str) -> Tuple[Optional[int], str]:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30, context=ssl_context()) as r:
            n = r.headers.get("Content-Length")
            return (int(n) if n else None), f"HTTP {r.status}"
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}"
    except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
        return None, _reason(e)


def check_urls(selected: List[Dataset]) -> int:
    """Send a HEAD request to every source and compare the reported size; nothing is downloaded.
    A file passes when at least one of its sources answers with the recorded size."""
    differs = unreachable = False
    for ds in selected:
        print(f"{ds.name}")
        for f in ds.files:
            verdicts = []
            for url in f.urls:
                n, status = _head(url)
                verdict = ("ok" if n == f.size else "SIZE DIFFERS" if n is not None
                           else "size unknown" if status.startswith("HTTP 2") else "UNREACHABLE")
                verdicts.append(verdict)
                print(f"  {verdict:12s} {status:8s} {url}")
            if not any(v in ("ok", "size unknown") for v in verdicts):
                differs = differs or "SIZE DIFFERS" in verdicts
                unreachable = unreachable or "SIZE DIFFERS" not in verdicts
        if ds.tree and ds.tree.url.startswith("https://github.com/"):
            owner_repo = ds.tree.url[len("https://github.com/"):].rstrip("/")
            owner_repo = owner_repo[:-4] if owner_repo.endswith(".git") else owner_repo
            api = f"https://api.github.com/repos/{owner_repo}/commits/{ds.tree.commit}"
            _, status = _head(api)
            ok = status == "HTTP 200"
            print(f"  {'ok' if ok else 'NO COMMIT':12s} {status:8s} {api}")
            unreachable = unreachable or not ok
    return 1 if differs else 3 if unreachable else 0


def fetch(selected: List[Dataset], root: Path, local: Optional[Path]) -> int:
    print(f"destination: {root}")
    print(f"source:      {'local tree ' + str(local) + ' (no network)' if local else 'the original public sources'}")
    tally = {"present": 0, "fetched": 0, "copied": 0, "mismatch": 0, "missing": 0}
    manual = []
    for ds in selected:
        print(f"\n{ds.name}: {ds.title}")
        print(f"  terms: {ds.terms}")
        failed = []
        for f in ds.files:
            r = obtain_file(f, root, local)
            tally[r.status] += 1
            print(f"  {r.status:8s} {r.shown:23s} {f.dest}")
            for p in r.problems:
                print(f"           {p}")
            if r.status in ("mismatch", "missing"):
                failed.append(f)
        if ds.tree:
            r = obtain_tree(ds.tree, root, local)
            tally[r.status] += 1
            print(f"  {r.status:8s} {r.shown:23s} {ds.tree.dest}/ (commit {ds.tree.commit[:12]})")
            for p in r.problems:
                print(f"           {p}")
            if r.status in ("mismatch", "missing"):
                failed.append(ds.tree)
        if failed:
            manual.append((ds, failed))

    ok = tally["present"] + tally["fetched"] + tally["copied"]
    print(f"\nsummary: {ok} verified ({tally['fetched']} fetched, {tally['copied']} copied, "
          f"{tally['present']} already present), {tally['mismatch']} checksum mismatches, "
          f"{tally['missing']} not obtained")
    for ds, failed in manual:
        print(f"\nMANUAL STEP for {ds.name} ({ds.source}):")
        if ds.manual:
            print(f"  {ds.manual}")
        for item in failed:
            if isinstance(item, File):
                print(f"  save {' or '.join(item.urls)}\n    as {root / item.dest}  ({expected(item)})")
        print("  then rerun this script to verify.")
    for d in DATASETS:
        if not d.default and d.name not in {s.name for s in selected}:
            print(f"\nnot fetched: {d.name} is optional; fetch it with --only {d.name}")
    return 1 if tally["mismatch"] else 3 if tally["missing"] else 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 epilog="datasets: " + ", ".join(NAMES) + ".  Exit status: 0 verified, "
                                        "1 checksum mismatch, 3 not obtained (manual step printed), 2 usage.")
    ap.add_argument("--list", action="store_true", help="show datasets, sources, terms and destinations; fetch nothing")
    ap.add_argument("--only", nargs="+", metavar="NAME", choices=NAMES, help="only these datasets")
    ap.add_argument("--from-local", metavar="DIR", type=Path,
                    help="copy from a directory laid out like materials/ instead of downloading, then verify")
    ap.add_argument("--root", metavar="DIR", type=Path,
                    help="repository root to write under (default: the parent of scripts/)")
    ap.add_argument("--check-urls", action="store_true",
                    help="send HEAD requests to every source and compare sizes; fetch nothing")
    args = ap.parse_args(argv)

    names = set(args.only) if args.only else {d.name for d in DATASETS if d.default}
    selected = [d for d in DATASETS if d.name in names]
    if args.list:
        list_datasets(selected)
        return 0
    if args.check_urls:
        return check_urls(selected)

    root = (args.root or REPO_ROOT).resolve()
    if args.root is None and not (root / "code").is_dir():
        ap.error(f"{root} has no code/ directory, so it does not look like the repository root; pass --root")
    local = None
    if args.from_local:
        local = args.from_local.expanduser().resolve()
        if not (local / "real_world_data").is_dir() and (local / "materials" / "real_world_data").is_dir():
            local = local / "materials"
        if not local.is_dir():
            ap.error(f"--from-local {local} is not a directory")
    root.mkdir(parents=True, exist_ok=True)
    return fetch(selected, root, local)


if __name__ == "__main__":
    sys.exit(main())
