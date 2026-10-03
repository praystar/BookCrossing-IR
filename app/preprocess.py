"""Text preprocessing: cleaning, tokenization, stop-word removal, stemming."""
import html
import re
import unicodedata
from functools import lru_cache

from nltk.stem import PorterStemmer

# NLTK's standard English list, embedded so no download is needed.
# (sklearn's list removes words like "fire", "bill", "system" -> bad for titles.)
STOP = frozenset("""i me my myself we our ours ourselves you your yours yourself yourselves he him his
himself she her hers herself it its itself they them their theirs themselves what which who whom this
that these those am is are was were be been being have has had having do does did doing a an the and
but if or because as until while of at by for with about against between into through during before
after above below to from up down in out on off over under again further then once here there when
where why how all any both each few more most other some such no nor not only own same so than too
very s t can will just don should now""".split())

_STEMMER = PorterStemmer()
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def clean(text) -> str:
    """Unescape HTML, strip accents, lowercase, keep only [a-z0-9] and spaces."""
    text = html.unescape(str(text if text is not None else ""))
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return _NON_ALNUM.sub(" ", text.lower()).strip()


@lru_cache(maxsize=500_000)
def stem(token: str) -> str:
    return _STEMMER.stem(token)


def tokenize(text) -> list[str]:
    return clean(text).split()


def preprocess(text, stop: bool = True, stemming: bool = True) -> list[str]:
    """Full pipeline. If stop-word removal would empty the text (e.g. the
    book "It"), the original tokens are kept."""
    toks = tokenize(text)
    if stop:
        kept = [t for t in toks if t not in STOP]
        toks = kept or toks
    if stemming:
        toks = [stem(t) for t in toks]
    return toks
