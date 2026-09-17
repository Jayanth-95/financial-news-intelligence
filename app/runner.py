import csv, logging, os, tempfile
from pathlib import Path
from .models import NewsItem
from .services import article_key
from utils.dates import is_within_window
LOG=logging.getLogger(__name__)
OUTPUT_COLUMNS=("source","url","published","title","summary","text","companies","tickers","mapped_tickers","percentages","currency_values","events","sentiment_label","sentiment_score")
class PipelineRunner:
    def __init__(self,sources,analyzer,output_path,days=1): self.sources=tuple(sources); self.analyzer=analyzer; self.output_path=Path(output_path); self.days=days
    def _load_existing_rows(self):
        """Return existing output rows, tolerating a missing/empty/malformed file.

        A row that doesn't resemble our schema (no source/title at all) or a
        file that can't be parsed as CSV is skipped rather than propagated,
        so a corrupt file degrades to "start fresh" instead of crashing the
        run. Rows are de-duplicated on load using the same article-key
        strategy used for newly collected items.
        """
        if not self.output_path.exists(): return []
        rows=[]; seen=set()
        try:
            with self.output_path.open(newline="",encoding="utf-8") as f:
                reader=csv.DictReader(f)
                if not reader.fieldnames: return []
                for r in reader:
                    source=(r.get("source") or "").strip(); title=(r.get("title") or "").strip()
                    if not source and not title: continue
                    key=article_key(NewsItem(source,title))
                    if key in seen: continue
                    seen.add(key)
                    rows.append({col:(r.get(col) or "") for col in OUTPUT_COLUMNS})
        except Exception as exc:
            LOG.warning("Existing output at %s is unreadable; starting from an empty dataset: %s",self.output_path,exc)
            return []
        return rows
    def _write_atomic(self,rows):
        self.output_path.parent.mkdir(parents=True,exist_ok=True)
        fd,tmp_path=tempfile.mkstemp(prefix=".tmp-",suffix=".csv",dir=str(self.output_path.parent))
        try:
            with os.fdopen(fd,"w",newline="",encoding="utf-8") as f:
                w=csv.DictWriter(f,fieldnames=OUTPUT_COLUMNS); w.writeheader(); w.writerows(rows)
            os.replace(tmp_path,self.output_path)
        except Exception:
            try: os.remove(tmp_path)
            except OSError: pass
            raise
    def execute(self):
        existing_rows=self._load_existing_rows()
        seen={article_key(NewsItem(r["source"],r["title"])) for r in existing_rows}
        new_rows=[]
        for i,source in enumerate(self.sources,1):
            name=getattr(source,"__name__",f"source-{i}"); LOG.info("Collecting %d/%d: %s",i,len(self.sources),name)
            try:
                for raw in source(days=self.days):
                    item=NewsItem.from_mapping(raw)
                    if not is_within_window(item.published, self.days):
                        LOG.debug("Skipping out-of-window article: %s (published=%r)",item.title,item.published)
                        continue
                    key=article_key(item)
                    if key in seen: continue
                    seen.add(key)
                    try:
                        result=self.analyzer.analyze(item)
                        if result: new_rows.append(result.as_row())
                    except Exception: LOG.exception("Article analysis failed: %s",item.title)
            except Exception: LOG.exception("Source failed: %s",name)
        all_rows=existing_rows+new_rows
        self._write_atomic(all_rows)
        LOG.info("Wrote %d rows to %s (%d existing preserved, %d new)",len(all_rows),self.output_path,len(existing_rows),len(new_rows))
        return len(all_rows)
