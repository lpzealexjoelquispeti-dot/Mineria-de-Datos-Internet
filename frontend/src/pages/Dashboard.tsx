import { useState } from "react";
import { Activity, Database, RadioTower, Router, WifiOff } from "lucide-react";

import { AreaChart } from "../components/AreaChart";
import { DistributionChart } from "../components/DistributionChart";
import { FindingsSection } from "../components/FindingsSection";
import { Header } from "../components/Header";
import { KpiCard } from "../components/KpiCard";
import { OutliersSection } from "../components/OutliersSection";
import { OverviewChart } from "../components/OverviewChart";
import { RankingChart } from "../components/RankingChart";
import { EmptyState, ErrorState, LoadingState } from "../components/States";
import { useDashboard } from "../hooks/useDashboard";
import type { AreaFilter, MetricFilter } from "../types/api";
import { formatDate, formatNumber } from "../utils/format";


const METRIC_LABELS: Record<MetricFilter, string> = {
  internet: "algún Internet",
  fijo: "Internet fijo",
  movil: "Internet móvil",
};


export default function Dashboard() {
  const [area, setArea] = useState<AreaFilter>("todos");
  const [metric, setMetric] = useState<MetricFilter>("internet");
  const { data, loading, error, reload } = useDashboard(area, metric);

  return (
    <main>
      <Header area={area} metric={metric} onAreaChange={setArea} onMetricChange={setMetric} />

      {loading && <LoadingState />}
      {error && <ErrorState message={error} onRetry={reload} />}
      {!loading && !error && !data && <EmptyState message="No hay resultados procesados disponibles." />}

      {data && (
        <>
          <section className="kpi-grid" aria-label="Indicadores principales">
            <KpiCard
              label="Registros analizados"
              value={data.resumen.total_registros}
              description={`${formatNumber(data.resumen.registros_validos)} en el universo TIC`}
              icon={Database}
              tone="navy"
            />
            <KpiCard label="Internet fijo" value={data.resumen.internet_fijo.cantidad} percentage={data.resumen.internet_fijo.porcentaje} icon={Router} tone="cyan" />
            <KpiCard label="Internet móvil" value={data.resumen.internet_movil.cantidad} percentage={data.resumen.internet_movil.porcentaje} icon={RadioTower} tone="amber" />
            <KpiCard label="Con algún Internet" value={data.resumen.algun_internet.cantidad} percentage={data.resumen.algun_internet.porcentaje} icon={Activity} tone="navy" />
            <KpiCard label="Sin Internet" value={data.resumen.sin_internet.cantidad} percentage={data.resumen.sin_internet.porcentaje} icon={WifiOff} tone="coral" />
          </section>

          <OverviewChart resumen={data.resumen} />

          <div className="dashboard-grid dashboard-grid--analytics">
            <AreaChart areas={data.areas} />
            <DistributionChart distribution={data.distribucion} />
          </div>

          <div className="rankings-heading">
            <div>
              <p className="section-number">RANKING TERRITORIAL</p>
              <h2>Municipios por {METRIC_LABELS[metric]}</h2>
            </div>
            <p>{area === "todos" ? "Todo el departamento" : `Área ${area}`} · {data.top.total} municipios con observaciones</p>
          </div>

          <div className="dashboard-grid dashboard-grid--rankings">
            <RankingChart number="04 · MAYOR CONECTIVIDAD" title="Los 10 con mayor acceso" items={data.top.items} metric={metric} color="#167f87" />
            <RankingChart number="05 · MENOR CONECTIVIDAD" title="Los 10 con menor acceso" items={data.bottom.items} metric={metric} color="#d56855" />
          </div>

          <OutliersSection outliers={data.outliers} />
          <FindingsSection findings={data.hallazgos} />

          <footer className="method-footer">
            <div>
              <span>UNIDAD DE ANÁLISIS</span>
              <strong>{data.metadata.unidad_analisis}</strong>
            </div>
            <div>
              <span>COBERTURA</span>
              <strong>{formatNumber(data.resumen.total_registros)} registros de La Paz</strong>
            </div>
            <div>
              <span>ÚLTIMA GENERACIÓN</span>
              <strong>{formatDate(data.metadata.generado_en)}</strong>
            </div>
            <p>{data.metadata.fuente}. Las asociaciones son descriptivas y no demuestran causalidad.</p>
          </footer>
        </>
      )}
    </main>
  );
}
