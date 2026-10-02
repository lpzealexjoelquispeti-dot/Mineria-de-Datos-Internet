import { SearchCheck } from "lucide-react";

import type { OutliersResponse } from "../types/api";
import { formatPercent } from "../utils/format";
import { EmptyState } from "./States";


export function OutliersSection({ outliers }: { outliers: OutliersResponse }) {
  return (
    <section className="outliers-section">
      <div className="outliers-intro">
        <p className="section-number">06 · LECTURA ESTADÍSTICA</p>
        <h2>Territorios atípicos</h2>
        <p>{outliers.descripcion}</p>
        <div className="method-pill">
          <SearchCheck size={17} aria-hidden="true" />
          Regla de Tukey · 1,5 × IQR
        </div>
      </div>
      <div className="outlier-cards">
        {outliers.items.length === 0 ? <EmptyState message="No se detectaron territorios atípicos." /> : outliers.items.map((item) => (
          <article className="outlier-card" key={item.codigo}>
            <span className="outlier-card__code">MUNICIPIO {item.codigo}</span>
            <strong>{item.nombre}</strong>
            <div className="outlier-card__value">{formatPercent(item.valor)}</div>
            <div className="outlier-card__scale" aria-hidden="true">
              <span style={{ width: `${Math.min(item.valor, 100)}%` }} />
            </div>
            <small>Límite superior: {formatPercent(item.limite_superior)}</small>
          </article>
        ))}
      </div>
    </section>
  );
}
