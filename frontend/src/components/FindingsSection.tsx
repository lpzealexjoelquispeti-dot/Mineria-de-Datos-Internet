import { ArrowUpRight, Lightbulb, Quote } from "lucide-react";

import type { HallazgosResponse } from "../types/api";


export function FindingsSection({ findings }: { findings: HallazgosResponse }) {
  return (
    <section className="findings-section">
      <div className="finding-main">
        <Quote size={24} aria-hidden="true" />
        <p className="section-number">07 · SÍNTESIS</p>
        <h2>Hallazgo principal</h2>
        <p>{findings.hallazgo_principal}</p>
      </div>
      <div className="finding-side">
        <div className="hypothesis-card">
          <Lightbulb size={21} aria-hidden="true" />
          <div>
            <h3>Hipótesis</h3>
            <p>{findings.hipotesis}</p>
          </div>
        </div>
        <div className="conclusions-card">
          <h3>Conclusiones principales</h3>
          <ol>
            {findings.conclusiones_principales.map((conclusion) => (
              <li key={conclusion}>
                <ArrowUpRight size={15} aria-hidden="true" />
                <span>{conclusion}</span>
              </li>
            ))}
          </ol>
        </div>
      </div>
    </section>
  );
}
