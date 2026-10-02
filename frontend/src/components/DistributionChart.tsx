import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { DistribucionResponse } from "../types/api";
import { buildHistogram, formatPercent } from "../utils/format";
import { Panel } from "./Panel";
import { EmptyState } from "./States";


export function DistributionChart({ distribution }: { distribution: DistribucionResponse }) {
  const histogram = buildHistogram(distribution.items.map((item) => item.valor));
  return (
    <Panel
      number="03 · DISPERSIÓN MUNICIPAL"
      title="Distribución de conectividad"
      note={<>Mediana: <strong>{formatPercent(distribution.mediana)}</strong></>}
      className="panel--chart"
    >
      {distribution.items.length === 0 ? <EmptyState /> : (
        <div className="chart chart--medium" aria-label="Distribución municipal de conectividad">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={histogram} margin={{ top: 18, right: 8, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 5" vertical={false} stroke="#d9e2ec" />
              <XAxis dataKey="label" tickLine={false} axisLine={false} tick={{ fontSize: 11 }} />
              <YAxis allowDecimals={false} tickLine={false} axisLine={false} />
              <Tooltip formatter={(value: number) => [`${value} municipios`, "Frecuencia"]} labelFormatter={(label) => `${label} %`} cursor={{ fill: "#edf4f7" }} />
              <Bar dataKey="cantidad" fill="#4fa9a5" radius={[5, 5, 0, 0]} maxBarSize={42} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </Panel>
  );
}
