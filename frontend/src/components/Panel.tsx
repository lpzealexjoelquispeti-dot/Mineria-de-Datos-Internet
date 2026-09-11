import type { PropsWithChildren, ReactNode } from "react";


interface PanelProps extends PropsWithChildren {
  number: string;
  title: string;
  note?: ReactNode;
  className?: string;
}


export function Panel({ number, title, note, className = "", children }: PanelProps) {
  return (
    <section className={`panel ${className}`.trim()}>
      <div className="panel__heading">
        <div>
          <p className="section-number">{number}</p>
          <h2>{title}</h2>
        </div>
        {note && <div className="panel__note">{note}</div>}
      </div>
      {children}
    </section>
  );
}
