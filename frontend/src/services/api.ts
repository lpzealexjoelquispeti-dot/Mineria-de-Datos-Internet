import type {
  AreaFilter,
  AreaResponse,
  DistribucionResponse,
  HallazgosResponse,
  LogisticRegressionResponse,
  MetadataResponse,
  MetricFilter,
  OrderFilter,
  OutliersResponse,
  Resumen,
  TerritoriosResponse,
} from "../types/api";


const API_URL = (import.meta.env.VITE_API_URL ?? "http://localhost:8000").replace(/\/$/, "");


async function request<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    headers: { Accept: "application/json" },
    signal,
  });
  if (!response.ok) {
    throw new Error(`La API respondió con estado ${response.status}.`);
  }
  return (await response.json()) as T;
}


export const api = {
  resumen: (signal?: AbortSignal) => request<Resumen>("/api/resumen", signal),
  areas: (signal?: AbortSignal) =>
    request<AreaResponse>("/api/conectividad/area", signal),
  municipios: (
    params: {
      limit: number;
      orden: OrderFilter;
      metrica: MetricFilter;
      area: AreaFilter;
    },
    signal?: AbortSignal,
  ) => {
    const query = new URLSearchParams({
      limit: String(params.limit),
      orden: params.orden,
      metrica: params.metrica,
      area: params.area,
    });
    return request<TerritoriosResponse>(`/api/conectividad/municipios?${query}`, signal);
  },
  distribucion: (
    metrica: MetricFilter,
    area: AreaFilter,
    signal?: AbortSignal,
  ) => {
    const query = new URLSearchParams({ metrica, area });
    return request<DistribucionResponse>(`/api/mineria/distribucion?${query}`, signal);
  },
  outliers: (signal?: AbortSignal) =>
    request<OutliersResponse>("/api/mineria/outliers", signal),
  hallazgos: (signal?: AbortSignal) =>
    request<HallazgosResponse>("/api/hallazgos", signal),
  metadata: (signal?: AbortSignal) =>
    request<MetadataResponse>("/api/metadata", signal),
  regresionLogistica: (signal?: AbortSignal) =>
    request<LogisticRegressionResponse>("/api/mineria/regresion-logistica", signal),
};
