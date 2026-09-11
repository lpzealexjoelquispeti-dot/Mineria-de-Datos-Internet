export interface Indicador {
  cantidad: number;
  porcentaje: number;
  porcentaje_determinados: number | null;
}

export interface Resumen {
  total_registros: number;
  registros_validos: number;
  registros_fuera_universo: number;
  internet_fijo: Indicador;
  internet_movil: Indicador;
  algun_internet: Indicador;
  sin_internet: Indicador;
  sin_especificar: Indicador;
}

export type AreaFilter = "todos" | "urbana" | "rural";
export type MetricFilter = "internet" | "fijo" | "movil";
export type OrderFilter = "mayor" | "menor";

export interface AreaConectividad {
  area: "Urbana" | "Rural";
  total: number;
  internet_fijo: Indicador;
  internet_movil: Indicador;
  algun_internet: Indicador;
  sin_internet: Indicador;
  sin_especificar: Indicador;
}

export interface AreaResponse {
  items: AreaConectividad[];
  brecha_urbano_rural_pp: number;
  v_cramer: number;
}

export interface TerritorioConectividad {
  codigo: string;
  nombre: string | null;
  total: number;
  internet_fijo: Indicador;
  internet_movil: Indicador;
  algun_internet: Indicador;
  sin_internet: Indicador;
  sin_especificar: Indicador;
}

export interface TerritoriosResponse {
  items: TerritorioConectividad[];
  total: number;
  limit: number;
  orden: OrderFilter;
  metrica: MetricFilter | "sin_internet";
  area: AreaFilter;
}

export interface Outlier {
  unidad_geografica: "Municipio";
  codigo: string;
  nombre: string;
  metrica: string;
  valor: number;
  metodo: string;
  q1: number;
  q3: number;
  iqr: number;
  limite_inferior: number;
  limite_superior: number;
}

export interface OutliersResponse {
  descripcion: string;
  items: Outlier[];
}

export interface DistribucionItem {
  codigo: string;
  nombre: string;
  valor: number;
}

export interface DistribucionResponse {
  metrica: MetricFilter;
  area: AreaFilter;
  items: DistribucionItem[];
  mediana: number;
}

export interface HallazgosResponse {
  hallazgo_principal: string;
  hipotesis: string;
  conclusiones_principales: string[];
}

export interface VariableMetadata {
  nombre: string;
  descripcion: string;
  codigos_o_rango: string;
  fuente: string;
}

export interface MetadataResponse {
  fuente: string;
  censo: string;
  departamento: string;
  unidad_analisis: string;
  variables_principales: string[];
  descripcion_variables: VariableMetadata[];
  generado_en: string;
  proceso: string;
}
