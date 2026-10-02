import { useCallback, useEffect, useState } from "react";
import { api } from "../services/api";
import type { RegressionTreeResponse } from "../types/api";

export function useRegressionTree() {
  const [model, setModel] = useState<RegressionTreeResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [requestId, setRequestId] = useState(0);
  const reload = useCallback(() => setRequestId((value) => value + 1), []);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    api.arbolRegresion(controller.signal)
      .then((result) => { if (!controller.signal.aborted) setModel(result); })
      .catch((cause: unknown) => {
        if (controller.signal.aborted) return;
        setError(cause instanceof Error ? cause.message : "No se pudo cargar H3_3.");
      })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [requestId]);
  return { model, loading, error, reload };
}
