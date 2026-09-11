import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { MetricFilter, TerritorioConectividad } from "../types/api";
import { formatPercent, shortName } from "../utils/format";
import { Panel } from "./Panel";
import { EmptyState } from "./States";


const metricValue = (item: TerritorioConectividad, metric: MetricFilter) => {
  if (metric === "fijo") return item.internet_fijo.porcentaje;
  if (metric === "movil") return item.internet_movil.porcentaje;
  return item.algun_internet.porcentaje;
};


interface RankingChartProps {
  title: string;
  number: string;
  items: TerritorioConectividad[];
  metric: MetricFilter;
  color: string;
}


export function RankingChart({ title, number, items, metric, color }: RankingChartProps) {
  const data = items.map((item) => ({
    name: shortName(item.nombre),
    fullName: item.nombre,
    porcentaje: metricValue(item, metric),
  }));

  return (
    <Panel number={number} title={title} className="panel--chart ranking-panel">
      {data.length === 0 ? <EmptyState /> : (
        <div className="chart chart--ranking" aria-label={title}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart layout="vertical" data={data} margin={{ top: 8, right: 50, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 5" horizontal={false} stroke="#e2e8ee" />
              <XAxis type="number" domain={[0, 100]} tickFormatter={(value: number) => `${value}%`} tickLine={false} axisLine={false} />
              <YAxis type="category" dataKey="name" width={142} tickLine={false} axisLine={false} tick={{ fill: "#43556d", fontSize: 12 }} />
              <Tooltip
                formatter={(value: number) => formatPercent(value)}
                labelFormatter={(_, payload) => payload[0]?.payload.fullName ?? ""}
                cursor={{ fill: "#edf4f7" }}
              />
              <Bar dataKey="porcentaje" fill={color} radius={[0, 5, 5, 0]} maxBarSize={20}>
                <LabelList dataKey="porcentaje" position="right" formatter={(value: number) => formatPercent(value)} fill="#263951" fontSize={11} fontWeight={700} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </Panel>
  );
}
