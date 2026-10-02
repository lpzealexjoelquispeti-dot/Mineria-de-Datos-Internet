import { Activity, DatabaseZap } from "lucide-react";


export function LoadingState() {
  return (
    <div className="state-card state-card--loading" aria-live="polite">
      <div className="loader" aria-hidden="true" />
      <div>
        <strong>Consultando resultados procesados</strong>
        <p>La API está preparando los indicadores del Censo 2024.</p>
      </div>
    </div>
  );
}


export function ErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="state-card state-card--error" role="alert">
      <Activity aria-hidden="true" />
      <div>
        <strong>No fue posible conectar con FastAPI</strong>
        <p>{message} Compruebe que el backend esté activo en el puerto 8000.</p>
      </div>
      <button type="button" onClick={onRetry}>Reintentar</button>
    </div>
  );
}


export function EmptyState({ message = "No hay datos para esta selección." }: { message?: string }) {
  return (
    <div className="empty-state">
      <DatabaseZap aria-hidden="true" />
      <p>{message}</p>
    </div>
  );
}
