class IncrementalMerger:
    def __init__(self, max_gap=0.5):
        self.max_gap = max_gap
        # Для каждого спикера храним текущий незавершённый буфер
        # speaker -> {"seg": [start,end], "texts": [..]}
        self._tails = {}

    def _merge_for_speaker(self, spk, entries):
        """
        Склеиваем новые entries для одного спикера, учитывая хвост.
        Возвращаем: завершённые фрагменты, плюс обновлённый хвост.
        """
        # Добавляем незавершённый хвост в начало
        tail = self._tails.get(spk)
        if tail:
            entries = [{"segment": tuple(tail["seg"]), "text": tail["texts"][-1]}] + entries
        # Сортировка по времени
        entries = sorted(entries, key=lambda e: e["segment"])
        completed = []
        cur_seg, cur_texts = None, []
        for ent in entries:
            s, e = ent["segment"]
            text = ent["text"]
            if cur_seg is None:
                cur_seg, cur_texts = [s, e], [text]
            else:
                if s <= cur_seg + self.max_gap:
                    # продолжаем
                    cur_seg = max(cur_seg, e)
                    cur_texts.append(text)
                else:
                    # фрагмент завершён
                    completed.append({
                        "segment": tuple(cur_seg),
                        "text": " ".join(cur_texts)
                    })
                    cur_seg, cur_texts = [s, e], [text]
        # После прохода последний cur_seg уходит в хвост
        if cur_seg:
            self._tails[spk] = {"seg": cur_seg, "texts": cur_texts}
        else:
            self._tails.pop(spk, None)
        return completed

    def merge_incremental(self, new_transcripts):
        """
        new_transcripts: speaker -> list of {"segment":(s,e), "text":str}
        Возвращает: speaker -> list завершённых фрагментов
        """
        results = {}
        for spk, entries in new_transcripts.items():
            completed = self._merge_for_speaker(spk, entries)
            if completed:
                results[spk] = completed
        return results
