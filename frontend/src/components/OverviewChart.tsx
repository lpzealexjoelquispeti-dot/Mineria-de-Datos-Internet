import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { Resumen } from "../types/api";
import { formatNumber, formatPercent } from "../utils/format";
import { Panel } from "./Panel";


const COLORS = ["#167f87", "#e3a13f", "#112744", "#d56855"];


export function OverviewChart({ resumen }: { resumen: Resumen }) {
  const data = [
    { name: "Internet fijo", porcentaje: resumen.internet_fijo.porcentaje },
    { name: "Internet móvil", porcentaje: resumen.internet_movil.porcentaje },
    { name: "Algún Internet", porcentaje: resumen.algun_internet.porcentaje },
    { name: "Sin Internet", porcentaje: resumen.sin_internet.porcentaje },
  ];

  return (
    <Panel
      number="01 · PANORAMA DE ACCESO"
      title="Internet fijo frente a Internet móvil"
      note={<>Porcentaje sobre {formatNumber(resumen.registros_validos)} viviendas aplicables</>}
      className="panel--wide"
    >
      <div className="chart chart--summary" aria-label="Comparación de acceso a Internet">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 24, right: 10, left: -18, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 5" vertical={false} stroke="#d9e2ec" />
            <XAxis dataKey="name" tickLine={false} axisLine={false} tick={{ fill: "#43556d", fontSize: 13 }} />
            <YAxis domain={[0, 100]} tickFormatter={(value: number) => `${value}%`} tickLine={false} axisLine={false} />
            <Tooltip formatter={(value: number) => formatPercent(value)} cursor={{ fill: "#edf4f7" }} />
            <Bar dataKey="porcentaje" radius={[7, 7, 0, 0]} maxBarSize={72}>
              {data.map((entry, index) => <Cell key={entry.name} fill={COLORS[index]} />)}
              <LabelList dataKey="porcentaje" position="top" formatter={(value: number) => formatPercent(value)} fill="#112744" fontSize={13} fontWeight={700} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Panel>
  );
}
