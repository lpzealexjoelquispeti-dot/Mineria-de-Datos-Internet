import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { LogisticMetrics, LogisticRegressionResponse } from "../types/api";
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


export function LogisticRegressionSection({ model }: { model: LogisticRegressionResponse }) {
  const matrix = model.matriz_confusion_test;
  const coefficients = model.coeficientes_principales.slice(0, 8);

  return (
    <section className="model-section" aria-labelledby="logistic-title">
      <div className="model-hero">
        <div>
          <p className="section-number">07 · MODELO PREDICTIVO</p>
          <h2 id="logistic-title">Regresión Logística</h2>
          <p>{model.objetivo}</p>
        </div>
        <div className="model-hero__facts">
          <span>Target</span>
          <strong>{model.target.nombre}</strong>
          <small>{model.target.variable_fuente} · clase positiva = {model.target.clase_positiva}</small>
        </div>
      </div>

      <div className="model-metrics" aria-label="Métricas del conjunto de prueba">
        {METRICS.map(({ key, label }) => (
          <article key={key}>
            <span>{label}</span>
            <strong>{percent(model.metricas.test[key])}</strong>
            <small>Test · umbral {model.umbral_principal.toFixed(1)}</small>
          </article>
        ))}
      </div>

      <div className="model-context">
        <p>{model.interpretacion_primera_iteracion}</p>
        <dl>
          <div><dt>Registros válidos</dt><dd>{formatNumber(model.registros.registros_target_valido)}</dd></div>
          <div><dt>Train / Test</dt><dd>{formatNumber(model.particion.train)} / {formatNumber(model.particion.test)}</dd></div>
          <div><dt>Clase positiva</dt><dd>{model.distribucion_clases_total.porcentaje_clase_positiva.toFixed(1).replace(".", ",")} %</dd></div>
          <div><dt>Baseline accuracy</dt><dd>{percent(model.baseline.accuracy_test)}</dd></div>
        </dl>
      </div>

      <div className="dashboard-grid model-grid">
        <article className="model-card">
          <div className="model-card__heading">
            <div>
              <p className="section-number">MATRIZ DE CONFUSIÓN · TEST</p>
              <h3>Predicciones a umbral 0,5</h3>
            </div>
            <span>{formatNumber(model.particion.test)} casos</span>
          </div>
          <div className="confusion-wrap">
            <div className="confusion-axis confusion-axis--top">Predicción</div>
            <div className="confusion-axis confusion-axis--side">Valor real</div>
            <div className="confusion-matrix">
              <div className="confusion-label" />
              <div className="confusion-label">Sin Internet</div>
              <div className="confusion-label">Con Internet</div>
              <div className="confusion-label">Sin Internet</div>
              <div className="confusion-cell confusion-cell--correct"><strong>{formatNumber(matrix.tn)}</strong><span>TN</span></div>
              <div className="confusion-cell confusion-cell--error"><strong>{formatNumber(matrix.fp)}</strong><span>FP</span></div>
              <div className="confusion-label">Con Internet</div>
              <div className="confusion-cell confusion-cell--error"><strong>{formatNumber(matrix.fn)}</strong><span>FN</span></div>
              <div className="confusion-cell confusion-cell--correct"><strong>{formatNumber(matrix.tp)}</strong><span>TP</span></div>
            </div>
          </div>
        </article>

        <article className="model-card">
          <div className="model-card__heading">
            <div>
              <p className="section-number">CURVA ROC</p>
              <h3>Capacidad de discriminación</h3>
            </div>
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
                <Line type="monotone" dataKey="tpr" name="TPR" stroke="#167f87" strokeWidth={3} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </article>
      </div>

      <div className="dashboard-grid model-grid">
        <article className="model-card">
          <div className="model-card__heading">
            <div>
              <p className="section-number">TRAIN VS TEST</p>
              <h3>Consistencia del modelo</h3>
            </div>
          </div>
          <div className="metric-table" role="table" aria-label="Comparación train y test">
            <div className="metric-table__row metric-table__head" role="row"><span>Métrica</span><span>Train</span><span>Test</span></div>
            {METRICS.map(({ key, label }) => (
              <div className="metric-table__row" role="row" key={key}>
                <strong>{label}</strong>
                <span>{percent(model.metricas.train[key])}</span>
                <span>{percent(model.metricas.test[key])}</span>
              </div>
            ))}
          </div>
        </article>

        <article className="model-card">
          <div className="model-card__heading">
            <div>
              <p className="section-number">VARIABLES EXPLICATIVAS</p>
              <h3>Características censales utilizadas</h3>
            </div>
          </div>
          <div className="variable-list">
            {model.variables_utilizadas.map((item) => (
              <div key={item.variable} title={item.motivo}>
                <code>{item.variable}</code>
                <span>{item.descripcion}</span>
              </div>
            ))}
          </div>
          <p className="leakage-note">{model.target.nota_leakage}</p>
        </article>
      </div>

      <article className="model-card model-card--associations">
        <div className="model-card__heading">
          <div>
            <p className="section-number">COEFICIENTES Y ODDS RATIOS</p>
            <h3>Asociaciones de mayor magnitud</h3>
          </div>
          <span>{model.statsmodels.variables_significativas_0_05.length} términos con p &lt; 0,05</span>
        </div>
        <div className="association-table">
          <div className="association-row association-row--head"><span>Variable transformada</span><span>Coeficiente</span><span>Odds Ratio</span><span>Asociación</span></div>
          {coefficients.map((item) => (
            <div className="association-row" key={item.feature}>
              <span>{item.descripcion}</span>
              <strong>{item.coeficiente.toFixed(3)}</strong>
              <strong>{item.odds_ratio.toFixed(3)}</strong>
              <span className={item.coeficiente >= 0 ? "association-positive" : "association-negative"}>
                {item.coeficiente >= 0 ? "Positiva" : "Negativa"}
              </span>
            </div>
          ))}
        </div>
        <p className="model-disclaimer">Los coeficientes expresan asociaciones condicionadas por las demás variables del modelo. No demuestran causalidad.</p>
      </article>
    </section>
  );
}
