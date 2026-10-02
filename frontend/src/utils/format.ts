export const formatNumber = new Intl.NumberFormat("es-BO").format;

export function formatPercent(value: number): string {
  return `${value.toLocaleString("es-BO", {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  })} %`;
}

export function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return value;
  return new Intl.DateTimeFormat("es-BO", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

export function shortName(value: string | null, length = 22): string {
  if (!value) return "Código sin nombre";
  return value.length > length ? `${value.slice(0, length - 1)}…` : value;
}

export interface HistogramBin {
  label: string;
  cantidad: number;
}

export function buildHistogram(values: number[]): HistogramBin[] {
  const bins = Array.from({ length: 10 }, (_, index) => ({
    label: `${index * 10}–${(index + 1) * 10}`,
    cantidad: 0,
  }));
  for (const value of values) {
    const index = Math.min(9, Math.max(0, Math.floor(value / 10)));
    const bin = bins[index];
    if (bin) bin.cantidad += 1;
  }
  return bins;
}
