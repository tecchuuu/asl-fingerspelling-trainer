/**
 * The word being spelled. statuses (optional) colors each letter:
 * "first" right first try, "retry" right after misses, "failed", or "done" (practice).
 */
export default function WordDisplay({ item, index, statuses }) {
  return (
    <div className="word">
      {[...item].map((c, i) => {
        const status = statuses?.[i] ?? (i < index ? "done" : null);
        return (
          <span key={i} className={status ?? (i === index ? "current" : "")}>{c}</span>
        );
      })}
    </div>
  );
}
