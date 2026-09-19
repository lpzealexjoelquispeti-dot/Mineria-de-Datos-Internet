import type { LucideIcon } from "lucide-react";

import { formatNumber, formatPercent } from "../utils/format";


interface KpiProps {
  label: string;
  value: number;
  percentage?: number;
  description?: string;
  icon: LucideIcon;
  tone: "navy" | "cyan" | "amber" | "coral";
}


export function KpiCard({ label, value, percentage, description, icon: Icon, tone }: KpiProps) {
  return (
    <article className={`kpi kpi--${tone}`}>
      <div className="kpi__topline">
        <span>{label}</span>
        <Icon aria-hidden="true" size={19} />
      </div>
      <strong>{formatNumber(value)}</strong>
      {percentage === undefined ? (
        <small>{description ?? "viviendas del universo TIC"}</small>
      ) : (
        <small>{formatPercent(percentage)} del universo</small>
      )}
    </article>
  );
}
