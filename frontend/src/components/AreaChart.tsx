import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { AreaResponse } from "../types/api";
import { formatPercent } from "../utils/format";
import { Panel } from "./Panel";
import { EmptyState } from "./States";


export function AreaChart({ areas }: { areas: AreaResponse }) {
  const data = areas.items.map((item) => ({
    area: item.area,
    fijo: item.internet_fijo.porcentaje,
    movil: item.internet_movil.porcentaje,
    internet: item.algun_internet.porcentaje,
    sin: item.sin_internet.porcentaje,
  }));

  return (
    <Panel
      number="02 · BRECHA TERRITORIAL"
      title="Urbano frente a rural"
      note={<><strong>{formatPercent(areas.brecha_urbano_rural_pp)}</strong> de brecha en algún Internet</>}
      className="panel--chart"
    >
      {data.length === 0 ? <EmptyState /> : (
        <div className="chart chart--medium" aria-label="Comparación urbana y rural">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 18, right: 4, left: -22, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 5" vertical={false} stroke="#d9e2ec" />
              <XAxis dataKey="area" tickLine={false} axisLine={false} />
              <YAxis domain={[0, 100]} tickFormatter={(value: number) => `${value}%`} tickLine={false} axisLine={false} />
              <Tooltip formatter={(value: number) => formatPercent(value)} cursor={{ fill: "#edf4f7" }} />
              <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12 }} />
              <Bar name="Fijo" dataKey="fijo" fill="#167f87" radius={[4, 4, 0, 0]} />
              <Bar name="Móvil" dataKey="movil" fill="#e3a13f" radius={[4, 4, 0, 0]} />
              <Bar name="Algún Internet" dataKey="internet" fill="#112744" radius={[4, 4, 0, 0]} />
              <Bar name="Sin Internet" dataKey="sin" fill="#d56855" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </Panel>
  );
}
