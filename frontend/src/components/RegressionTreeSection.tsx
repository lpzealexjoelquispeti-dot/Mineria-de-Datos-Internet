import {
  Bar, BarChart, CartesianGrid, ReferenceLine, ResponsiveContainer,
  Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from "recharts";

import { useRegressionTree } from "../hooks/useRegressionTree";
import type { RegressionMetrics } from "../types/api";
import { formatNumber, formatPercent } from "../utils/format";
import { LoadingState } from "./States";

const METRICS: Array<{ key: keyof RegressionMetrics; label: string; unit: string }> = [
  { key: "mae", label: "MAE", unit: "pp" },
  { key: "rmse", label: "RMSE", unit: "pp" },
  { key: "mse", label: "MSE", unit: "pp²" },
  { key: "r2", label: "R²", unit: "" },
];
const LABELS: Record<string, string> = {
  pct_urbano: "Viviendas urbanas", pct_con_energia: "Electricidad",
  pct_computadora: "Computadora", pct_celular: "Celular",
  promedio_habitaciones: "Habitaciones", promedio_personas: "Personas",
};
const decimal = (value: number) => value.toLocaleString("es-BO", { maximumFractionDigits: 3 });

export function RegressionTreeSection() {
  const { model, loading, error, reload } = useRegressionTree();
  return (
    <section className="model-section regression-section" aria-labelledby="regression-title">
      <div className="model-hero model-hero--regression">
        <div>
          <p className="section-number">09 · PREDICCIÓN TERRITORIAL · H3_3</p>
          <h2 id="regression-title">Árbol de Decisión — Regresión</h2>
          <p>Una observación = municipio. Predicción del porcentaje municipal de viviendas con acceso a Internet.</p>
        </div>
        {model && <div className="model-hero__facts">
          <span>Unidad de análisis</span>
          <strong>{formatNumber(model.registros.municipios)} municipios</strong>
          <small>Train / test: {model.particion.train} / {model.particion.test}</small>
        </div>}
      </div>
      {loading && <LoadingState />}
      {error && <div className="state-card state-card--error" role="alert">
        <div><strong>H3_3 no está disponible</strong><p>{error} No se pudieron consultar los resultados de regresión territorial.</p></div>
        <button type="button" onClick={reload}>Reintentar</button>
      </div>}
      {model && !loading && !error && <>
        <div className="model-metrics regression-metrics" aria-label="Métricas de regresión en test">
          {METRICS.map(({ key, label, unit }) => <article key={key}>
            <span>{label}</span><strong>{decimal(model.metricas.test[key])}</strong><small>Test {unit && `· ${unit}`}</small>
          </article>)}
        </div>
        <div className="model-context">
          <p>{model.interpretacion}</p>
          <dl>
            <div><dt>Municipios train / test</dt><dd>{model.particion.train} / {model.particion.test}</dd></div>
            <div><dt>Profundidad</dt><dd>{model.estructura_arbol.profundidad}</dd></div>
            <div><dt>Nodos / hojas</dt><dd>{model.estructura_arbol.nodos} / {model.estructura_arbol.hojas}</dd></div>
            <div><dt>Target</dt><dd>Porcentaje municipal · {model.target.indicador_eda}</dd></div>
          </dl>
        </div>
        <div className="dashboard-grid model-grid">
          <article className="model-card">
            <div className="model-card__heading"><div><p className="section-number">REAL VS. PREDICHO · TEST</p><h3>Porcentaje de acceso municipal</h3></div></div>
            <div className="regression-scatter" aria-label="Porcentaje real en X y predicho en Y; línea ideal y igual a x">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 20, right: 25, bottom: 30, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#d6e0e8" />
                  <XAxis dataKey="valor_real" type="number" name="Real" unit=" %" domain={[0, 100]} label={{ value: "Real (%)", position: "bottom", offset: 8 }} />
                  <YAxis dataKey="valor_predicho" type="number" name="Predicho" unit=" %" domain={[0, 100]} label={{ value: "Predicho (%)", angle: -90, position: "insideLeft" }} />
                  <Tooltip content={({ active, payload }) => {
                    const row = payload?.[0]?.payload;
                    return active && row ? <div className="regression-tooltip"><strong>{row.municipio}</strong><p>Real: {formatPercent(row.valor_real)}</p><p>Predicho: {formatPercent(row.valor_predicho)}</p><p>Error absoluto: {decimal(row.error_absoluto)} pp</p></div> : null;
                  }} />
                  <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 100, y: 100 }]} stroke="#b26128" strokeDasharray="5 5" />
                  <Scatter data={model.real_vs_predicho} name="Municipios test" fill="#167f87" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
            <p className="model-disclaimer">La línea discontinua representa y = x. Cada punto corresponde a un municipio de test.</p>
          </article>
          <article className="model-card">
            <div className="model-card__heading"><div><p className="section-number">IMPORTANCIA DE VARIABLES</p><h3>Contribución a la predicción</h3></div></div>
            <div className="importance-chart">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={model.importancia_variables.map((item) => ({ ...item, nombre: LABELS[item.variable] ?? item.variable }))} layout="vertical" margin={{ top: 10, right: 20, bottom: 10, left: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e1e7ec" />
                  <XAxis type="number" domain={[0, 1]} tickFormatter={(value) => `${(Number(value) * 100).toFixed(0)} %`} />
                  <YAxis type="category" dataKey="nombre" width={110} />
                  <Tooltip formatter={(value: number) => formatPercent(value * 100)} />
                  <Bar dataKey="importancia" name="Importancia" fill="#167f87" radius={[0, 5, 5, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="model-disclaimer">La importancia refleja el uso de cada predictor en este árbol y no implica causalidad.</p>
          </article>
        </div>
        <div className="dashboard-grid model-grid">
          <article className="model-card">
            <div className="model-card__heading"><div><p className="section-number">ÁRBOL VS. BASELINE · TEST</p><h3>Comparación con el promedio de train</h3></div></div>
            <div className="metric-table" role="table" aria-label="Árbol frente a DummyRegressor">
              <div className="metric-table__row metric-table__head" role="row"><span>Métrica</span><span>Árbol</span><span>Baseline</span></div>
              {METRICS.map(({ key, label, unit }) => <div className="metric-table__row" role="row" key={key}><strong>{label} {unit}</strong><span>{decimal(model.metricas.test[key])}</span><span>{decimal(model.baseline.metricas_test[key])}</span></div>)}
            </div>
            <p className="model-disclaimer">DummyRegressor predice {formatPercent(model.baseline.media_train)}, aprendido solo de train. Menor MAE/RMSE/MSE y mayor R² indican mejor rendimiento.</p>
            <h4>Train frente a test</h4>
            <div className="metric-table" role="table" aria-label="Regresión train y test">
              <div className="metric-table__row metric-table__head" role="row"><span>Métrica</span><span>Train</span><span>Test</span></div>
              {METRICS.map(({ key, label, unit }) => <div className="metric-table__row" role="row" key={key}><strong>{label} {unit}</strong><span>{decimal(model.metricas.train[key])}</span><span>{decimal(model.metricas.test[key])}</span></div>)}
            </div>
          </article>
          <article className="model-card">
            <div className="model-card__heading"><div><p className="section-number">MEJORES HIPERPARÁMETROS</p><h3>Selección mediante CV de train</h3></div><span>{model.gridsearch.cv} folds</span></div>
            <div className="tree-params">
              {Object.entries(model.gridsearch.mejores_parametros).map(([key, value]) => <div key={key}><code>{key}</code><strong>{value}</strong></div>)}
            </div>
            <p className="model-disclaimer">{model.gridsearch.combinaciones_evaluadas} configuraciones · {model.gridsearch.ajustes_cv} ajustes CV. MSE CV: {decimal(model.gridsearch.mejor_mse_cv)} pp²; desviación: {decimal(model.gridsearch.std_mse_cv)} pp².</p>
            <p className="model-disclaimer">Target mínimo {formatPercent(model.registros.estadisticos_target.minimo)}, máximo {formatPercent(model.registros.estadisticos_target.maximo)}, media {formatPercent(model.registros.estadisticos_target.promedio)} y mediana {formatPercent(model.registros.estadisticos_target.mediana)}.</p>
            <p className="leakage-note">{model.target.nota_leakage}</p>
          </article>
        </div>
        <article className="model-card model-card--comparison">
          <div className="model-card__heading"><div><p className="section-number">ERRORES MUNICIPALES · TEST</p><h3>Real, predicho y residuo</h3></div><span>{model.particion.test} municipios</span></div>
          <div className="regression-table-wrap"><table className="regression-table">
            <caption>Municipios ordenados por error absoluto. Residuo = real − predicho.</caption>
            <thead><tr><th scope="col">Municipio</th><th scope="col">Real (%)</th><th scope="col">Predicho (%)</th><th scope="col">Residuo (pp)</th><th scope="col">Error absoluto (pp)</th></tr></thead>
            <tbody>{model.real_vs_predicho.map((row) => <tr key={row.municipio_codigo}><th scope="row">{row.municipio} <small>{row.municipio_codigo}</small></th><td>{decimal(row.valor_real)}</td><td>{decimal(row.valor_predicho)}</td><td>{decimal(row.error)}</td><td>{decimal(row.error_absoluto)}</td></tr>)}</tbody>
          </table></div>
          <p className="model-disclaimer">Residuo promedio {decimal(model.residuos.promedio)} pp; mínimo {decimal(model.residuos.minimo)} pp; máximo {decimal(model.residuos.maximo)} pp.</p>
        </article>
        <article className="model-card model-card--comparison">
          <div className="model-card__heading"><div><p className="section-number">ALCANCE TERRITORIAL</p><h3>Limitaciones de H3_3</h3></div></div>
          <ul className="regression-limitations">{model.limitaciones.map((item) => <li key={item}>{item}</li>)}</ul>
          <p className="model-disclaimer">{model.particion.nota}</p>
        </article>
      </>}
    </section>
  );
}
