import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type {
  DecisionTreeResponse,
  LogisticMetrics,
  ModelComparison,
} from "../types/api";
import { formatNumber } from "../utils/format";


const METRICS: Array<{ key: keyof LogisticMetrics; label: string }> = [
  { key: "accuracy", label: "Accuracy" },
  { key: "precision", label: "Precision" },
  { key: "recall", label: "Recall" },
  { key: "f1", label: "F1" },
  { key: "roc_auc", label: "ROC-AUC" },
];


function percent(value: number) {
  return `${(value * 100).toFixed(1).replace(".", ",")} %`;
}


function comparisonRows(rows: ModelComparison[]) {
  return METRICS.map(({ key, label }) => ({
    metrica: label,
    "Regresión Logística": rows.find((row) => row.modelo === "Regresión Logística")?.[key] ?? 0,
    "Árbol de Decisión": rows.find((row) => row.modelo === "Árbol de Decisión")?.[key] ?? 0,
  }));
}


export function DecisionTreeSection({ model }: { model: DecisionTreeResponse }) {
  const matrix = model.matriz_confusion_test;
  const importance = model.importancia_variables.agregadas.slice(0, 7);
  const comparison = comparisonRows(model.comparacion_modelos);
  const params = model.gridsearch.mejores_parametros;

  return (
    <section className="model-section tree-section" aria-labelledby="tree-title">
      <div className="model-hero model-hero--tree">
        <div>
          <p className="section-number">08 · MODELO PREDICTIVO</p>
          <h2 id="tree-title">Árbol de Decisión — Clasificación</h2>
          <p>{model.objetivo}</p>
        </div>
        <div className="model-hero__facts">
          <span>Optimización</span>
          <strong>GridSearchCV · {model.gridsearch.cv} folds</strong>
          <small>ROC-AUC CV {model.gridsearch.mejor_roc_auc_cv.toFixed(3)}</small>
        </div>
      </div>

      <div className="model-metrics" aria-label="Métricas del árbol en prueba">
        {METRICS.map(({ key, label }) => (
          <article key={key}>
            <span>{label}</span>
            <strong>{percent(model.metricas.test[key])}</strong>
            <small>Test · umbral {model.umbral_estandar.toFixed(1)}</small>
          </article>
        ))}
      </div>

      <div className="model-context model-context--tree">
        <p>{model.interpretacion}</p>
        <dl>
          <div><dt>Profundidad</dt><dd>{model.estructura_arbol.profundidad}</dd></div>
          <div><dt>Nodos / hojas</dt><dd>{formatNumber(model.estructura_arbol.nodos)} / {formatNumber(model.estructura_arbol.hojas)}</dd></div>
          <div><dt>Umbral ROC</dt><dd>{model.seleccion_umbral_roc.umbral.toFixed(3)}</dd></div>
          <div><dt>Casos test</dt><dd>{formatNumber(model.particion.test)}</dd></div>
        </dl>
      </div>

      <div className="dashboard-grid model-grid">
        <article className="model-card">
          <div className="model-card__heading">
            <div><p className="section-number">MEJORES HIPERPARÁMETROS</p><h3>Complejidad seleccionada</h3></div>
            <span>{model.gridsearch.ajustes_cv} ajustes CV</span>
          </div>
          <div className="tree-params">
            <div><code>max_depth</code><strong>{params.max_depth}</strong></div>
            <div><code>min_samples_split</code><strong>{params.min_samples_split}</strong></div>
            <div><code>min_samples_leaf</code><strong>{params.min_samples_leaf}</strong></div>
          </div>
          <p className="model-disclaimer">{model.gridsearch.estrategia}</p>
        </article>

        <article className="model-card">
          <div className="model-card__heading">
            <div><p className="section-number">MATRIZ DE CONFUSIÓN · TEST</p><h3>Umbral estándar 0,5</h3></div>
          </div>
          <div className="confusion-wrap">
            <div className="confusion-axis confusion-axis--top">Predicción</div>
            <div className="confusion-axis confusion-axis--side">Valor real</div>
            <div className="confusion-matrix">
              <div className="confusion-label" /><div className="confusion-label">Sin Internet</div><div className="confusion-label">Con Internet</div>
              <div className="confusion-label">Sin Internet</div><div className="confusion-cell confusion-cell--correct"><strong>{formatNumber(matrix.tn)}</strong><span>TN</span></div><div className="confusion-cell confusion-cell--error"><strong>{formatNumber(matrix.fp)}</strong><span>FP</span></div>
              <div className="confusion-label">Con Internet</div><div className="confusion-cell confusion-cell--error"><strong>{formatNumber(matrix.fn)}</strong><span>FN</span></div><div className="confusion-cell confusion-cell--correct"><strong>{formatNumber(matrix.tp)}</strong><span>TP</span></div>
            </div>
          </div>
        </article>
      </div>

      <div className="dashboard-grid model-grid">
        <article className="model-card">
          <div className="model-card__heading">
            <div><p className="section-number">CURVA ROC</p><h3>Capacidad de discriminación</h3></div>
            <span>AUC {model.roc.auc_test.toFixed(3)}</span>
          </div>
          <div className="model-roc">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={model.roc.puntos} margin={{ top: 12, right: 18, bottom: 8, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#d6e0e8" />
                <XAxis dataKey="fpr" type="number" domain={[0, 1]} tickFormatter={(value) => Number(value).toFixed(1)} />
                <YAxis dataKey="tpr" type="number" domain={[0, 1]} tickFormatter={(value) => Number(value).toFixed(1)} />
                <Tooltip formatter={(value: number) => value.toFixed(3)} labelFormatter={(value) => `FPR ${Number(value).toFixed(3)}`} />
                <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#9aa9b6" strokeDasharray="5 5" />
                <Line type="stepAfter" dataKey="tpr" name="TPR" stroke="#b26128" strokeWidth={3} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="model-card">
          <div className="model-card__heading">
            <div><p className="section-number">IMPORTANCIA AGREGADA</p><h3>Variables originales</h3></div>
            <span>Suma {model.importancia_variables.suma_agregada.toFixed(3)}</span>
          </div>
          <div className="importance-chart">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={importance} layout="vertical" margin={{ top: 10, right: 24, bottom: 10, left: 12 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e1e7ec" />
                <XAxis type="number" domain={[0, "dataMax"]} tickFormatter={(value) => `${(Number(value) * 100).toFixed(0)} %`} />
                <YAxis type="category" dataKey="variable_original" width={105} />
                <Tooltip formatter={(value: number) => percent(value)} />
                <Bar dataKey="importancia" name="Importancia" fill="#b26128" radius={[0, 5, 5, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>
      </div>

      <article className="model-card model-card--comparison">
        <div className="model-card__heading">
          <div><p className="section-number">COMPARACIÓN DE MODELOS DE CLASIFICACIÓN</p><h3>Regresión Logística vs. Árbol de Decisión</h3></div>
          <span>Mismo test · {formatNumber(model.particion.test)} casos</span>
        </div>
        <div className="comparison-chart">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={comparison} margin={{ top: 20, right: 18, bottom: 8, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e1e7ec" />
              <XAxis dataKey="metrica" />
              <YAxis domain={[0, 1]} tickFormatter={(value) => `${(Number(value) * 100).toFixed(0)} %`} />
              <Tooltip formatter={(value: number) => percent(value)} />
              <Legend />
              <Bar dataKey="Regresión Logística" fill="#167f87" radius={[5, 5, 0, 0]} />
              <Bar dataKey="Árbol de Decisión" fill="#b26128" radius={[5, 5, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="model-disclaimer">La comparación usa los mismos registros, índices de test, target, variables y partición. Las barras presentan resultados; no establecen automáticamente un modelo ganador.</p>
      </article>
    </section>
  );
}
