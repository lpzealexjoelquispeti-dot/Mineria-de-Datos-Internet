import { useCallback, useEffect, useState } from "react";

import { api } from "../services/api";
import type {
  AreaFilter,
  AreaResponse,
  DecisionTreeResponse,
  DistribucionResponse,
  HallazgosResponse,
  LogisticRegressionResponse,
  MetadataResponse,
  MetricFilter,
  OutliersResponse,
  Resumen,
  TerritoriosResponse,
} from "../types/api";


export interface DashboardData {
  resumen: Resumen;
  areas: AreaResponse;
  top: TerritoriosResponse;
  bottom: TerritoriosResponse;
  distribucion: DistribucionResponse;
  outliers: OutliersResponse;
  hallazgos: HallazgosResponse;
  metadata: MetadataResponse;
  regresion: LogisticRegressionResponse;
  arbol: DecisionTreeResponse;
}

interface DashboardState {
  data: DashboardData | null;
  loading: boolean;
  error: string | null;
  reload: () => void;
}


export function useDashboard(area: AreaFilter, metric: MetricFilter): DashboardState {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [requestId, setRequestId] = useState(0);
  const reload = useCallback(() => setRequestId((value) => value + 1), []);

  useEffect(() => {
    const controller = new AbortController();
    const signal = controller.signal;
    setLoading(true);
    setError(null);

    Promise.all([
      api.resumen(signal),
      api.areas(signal),
      api.municipios({ limit: 10, orden: "mayor", metrica: metric, area }, signal),
      api.municipios({ limit: 10, orden: "menor", metrica: metric, area }, signal),
      api.distribucion(metric, area, signal),
      api.outliers(signal),
      api.hallazgos(signal),
      api.metadata(signal),
      api.regresionLogistica(signal),
      api.arbolClasificacion(signal),
    ])
      .then(([resumen, areas, top, bottom, distribucion, outliers, hallazgos, metadata, regresion, arbol]) => {
        setData({ resumen, areas, top, bottom, distribucion, outliers, hallazgos, metadata, regresion, arbol });
      })
      .catch((cause: unknown) => {
        if (cause instanceof DOMException && cause.name === "AbortError") return;
        setError(cause instanceof Error ? cause.message : "No se pudo cargar el dashboard.");
      })
      .finally(() => {
        if (!signal.aborted) setLoading(false);
      });

    return () => controller.abort();
  }, [area, metric, requestId]);

  return { data, loading, error, reload };
}
