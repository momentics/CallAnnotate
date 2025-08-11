import threading

class ThreadSafeIncrementalMerger:
    """
    Склеивает переходящие сегменты между окнами в многопоточном режиме.
    Для каждого спикера хранит незавершённый «хвост» буфера и возвращает только полностью завершённые фрагменты.
    """
    def __init__(self, max_gap: float = 0.5):
        self.max_gap = max_gap
        self._tails = {}          # speaker -> {"seg": [start,end], "texts": [..]}
        self._lock = threading.Lock()

    def merge(self, new: dict) -> dict:
        """
        new: speaker -> list of {"segment": (s,e), "text": str}
        Возвращает: speaker -> list завершённых {"segment":(s,e),"text":str}
        """
        completed = {}
        with self._lock:
            for spk, entries in new.items():
                # объединяем с хвостом
                tail = self._tails.get(spk)
                if tail:
                    entries = [{"segment": tuple(tail["seg"]), "text": tail["texts"][-1]}] + entries
                entries.sort(key=lambda e: e["segment"])

                cur_seg, cur_texts = None, []
                comp = []
                for ent in entries:
                    s, e = ent["segment"]
                    txt = ent["text"]
                    if cur_seg is None:
                        cur_seg, cur_texts = [s, e], [txt]
                    elif s <= cur_seg + self.max_gap:
                        cur_seg = max(cur_seg, e)
                        cur_texts.append(txt)
                    else:
                        comp.append({"segment": tuple(cur_seg), "text": " ".join(cur_texts)})
                        cur_seg, cur_texts = [s, e], [txt]
                # обновляем хвост
                if cur_seg:
                    self._tails[spk] = {"seg": cur_seg, "texts": cur_texts}
                else:
                    self._tails.pop(spk, None)

                if comp:
                    completed[spk] = comp
        return completed

