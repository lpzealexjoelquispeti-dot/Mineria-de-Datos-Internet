import { SlidersHorizontal } from "lucide-react";

import type { AreaFilter, MetricFilter } from "../types/api";


interface HeaderProps {
  area: AreaFilter;
  metric: MetricFilter;
  onAreaChange: (area: AreaFilter) => void;
  onMetricChange: (metric: MetricFilter) => void;
}


export function Header({ area, metric, onAreaChange, onMetricChange }: HeaderProps) {
  return (
    <>
      <header className="masthead">
        <div className="masthead__mark" aria-hidden="true">LP</div>
        <div className="masthead__copy">
          <p className="eyebrow">OBSERVATORIO DE CONECTIVIDAD · CENSO 2024</p>
          <h1>Conectividad Digital en La Paz</h1>
          <p>Análisis exploratorio del acceso a Internet fijo y móvil en viviendas del departamento.</p>
        </div>
        <div className="source-badge">
          <span className="source-badge__dot" />
          Datos procesados del Censo 2024
        </div>
      </header>

      <section className="filter-bar" aria-label="Filtros del análisis">
        <div className="filter-bar__title">
          <SlidersHorizontal size={18} aria-hidden="true" />
          <span>Explorar resultados</span>
        </div>
        <label>
          <span>Área</span>
          <select value={area} onChange={(event) => onAreaChange(event.target.value as AreaFilter)}>
            <option value="todos">Todos</option>
            <option value="urbana">Urbana</option>
            <option value="rural">Rural</option>
          </select>
        </label>
        <label>
          <span>Métrica</span>
          <select value={metric} onChange={(event) => onMetricChange(event.target.value as MetricFilter)}>
            <option value="internet">Algún Internet</option>
            <option value="fijo">Internet fijo</option>
            <option value="movil">Internet móvil</option>
          </select>
        </label>
        <p>Los filtros actualizan rankings y distribución municipal.</p>
      </section>
    </>
  );
}
