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

export interface LogisticMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
}

export interface LogisticVariable {
  variable: string;
  descripcion: string;
  tipo: string;
  motivo: string;
}

export interface LogisticCoefficient {
  feature: string;
  variable_original: string;
  categoria_codigo: string | null;
  categoria: string | null;
  descripcion: string;
  coeficiente: number;
  odds_ratio: number;
  impacto_absoluto: number;
  p_value: number | null;
}

export interface LogisticRegressionResponse {
  objetivo: string;
  target: {
    nombre: string;
    variable_fuente: string;
    descripcion_fuente: string;
    codificacion: Record<string, string>;
    clase_positiva: number;
    nota_leakage: string;
  };
  variables_utilizadas: LogisticVariable[];
  registros: {
    registros_lapaz: number;
    universo_tic: number;
    registros_target_valido: number;
    registros_excluidos: number;
    motivos_exclusion: Record<string, number>;
  };
  distribucion_clases_total: {
    "0_sin_internet": number;
    "1_con_internet": number;
    porcentaje_clase_positiva: number;
  };
  particion: {
    train: number;
    test: number;
    test_size: number;
    random_state: number;
    estratificada: boolean;
    distribucion_train: Record<string, number>;
    distribucion_test: Record<string, number>;
  };
  metricas: { train: LogisticMetrics; test: LogisticMetrics };
  matriz_confusion_test: {
    tn: number;
    fp: number;
    fn: number;
    tp: number;
    matriz: number[][];
  };
  baseline: {
    descripcion: string;
    clase_mayoritaria: number;
    accuracy_test: number;
  };
  umbral_principal: number;
  roc: {
    auc_test: number;
    puntos: Array<{ fpr: number; tpr: number; threshold: number | null }>;
  };
  coeficientes_principales: LogisticCoefficient[];
  statsmodels: {
    convergio: boolean;
    pseudo_r2_mcfadden: number;
    variables_significativas_0_05: unknown[];
  };
  interpretacion_primera_iteracion: string;
}

export interface TreeImportance {
  feature?: string;
  variable_original: string;
  categoria_codigo?: string | null;
  categoria?: string | null;
  importancia: number;
}

export interface ModelComparison extends LogisticMetrics {
  modelo: string;
}

export interface DecisionTreeResponse {
  objetivo: string;
  target: LogisticRegressionResponse["target"];
  variables_utilizadas: LogisticVariable[];
  registros: LogisticRegressionResponse["registros"];
  distribucion_clases_total: LogisticRegressionResponse["distribucion_clases_total"];
  particion: LogisticRegressionResponse["particion"];
  gridsearch: {
    cv: number;
    scoring: string;
    n_jobs: number;
    mejores_parametros: {
      max_depth: number;
      min_samples_split: number;
      min_samples_leaf: number;
    };
    mejor_roc_auc_cv: number;
    combinaciones_evaluadas: number;
    ajustes_cv: number;
    estrategia: string;
    motivo_reduccion: string;
  };
  umbral_estandar: number;
  metricas: { train: LogisticMetrics; test: LogisticMetrics };
  matriz_confusion_test: LogisticRegressionResponse["matriz_confusion_test"];
  classification_report_test: Record<string, unknown>;
  roc: LogisticRegressionResponse["roc"];
  seleccion_umbral_roc: {
    umbral: number;
    criterio: string;
    origen: string;
    indice_youden: number;
    metricas_train: LogisticMetrics;
    metricas_test: LogisticMetrics;
    matriz_confusion_test: LogisticRegressionResponse["matriz_confusion_test"];
  };
  importancia_variables: {
    transformadas: TreeImportance[];
    agregadas: TreeImportance[];
    suma_agregada: number;
  };
  estructura_arbol: {
    profundidad: number;
    nodos: number;
    hojas: number;
    visualizacion_max_depth: number;
    nota_visualizacion: string;
  };
  comparacion_modelos: ModelComparison[];
  verificacion_comparabilidad: Record<string, boolean>;
  interpretacion: string;
  artefactos: Record<string, string>;
}


export interface RegressionMetrics {
  mse: number;
  rmse: number;
  mae: number;
  r2: number;
}

export interface MunicipalPrediction {
  municipio_codigo: string;
  municipio: string;
  valor_real: number;
  valor_predicho: number;
  error: number;
  error_absoluto: number;
}

export interface RegressionTreeResponse {
  objetivo: string;
  unidad_analisis: "Municipio";
  target: { nombre: string; indicador_eda: string; definicion: string; nota_leakage: string };
  variables_utilizadas: Array<{ variable: string; descripcion: string }>;
  registros: {
    municipios: number;
    estadisticos_target: { minimo: number; maximo: number; promedio: number; mediana: number };
  };
  particion: { train: number; test: number; random_state: number; test_size: number; nota: string };
  gridsearch: {
    mejores_parametros: { max_depth: number; min_samples_split: number; min_samples_leaf: number };
    mejor_mse_cv: number;
    std_mse_cv: number;
    combinaciones_evaluadas: number;
    ajustes_cv: number;
    cv: number;
  };
  metricas: { train: RegressionMetrics; test: RegressionMetrics };
  baseline: { media_train: number; metricas_test: RegressionMetrics };
  importancia_variables: Array<{ variable: string; importancia: number }>;
  estructura_arbol: { profundidad: number; nodos: number; hojas: number };
  real_vs_predicho: MunicipalPrediction[];
  municipios_mayor_error: MunicipalPrediction[];
  residuos: { promedio: number; minimo: number; maximo: number; mae: number };
  interpretacion: string;
  limitaciones: string[];
}
