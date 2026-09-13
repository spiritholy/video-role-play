// 统一的请求封装：把 fetch 异常与非 2xx 收敛成结果对象，调用侧不再写 try/catch。
// 语义与拆分前一致——网络异常（含 2xx 但响应体不是合法 JSON）走
// NETWORK_MESSAGE，非 2xx 走后端 detail 或调用点自带的默认文案。

export type ApiOk<T> = { ok: true; data: T };
export type ApiErr = { ok: false; detail: string | null; network: boolean };
export type ApiResult<T> = ApiOk<T> | ApiErr;

export const NETWORK_MESSAGE = "无法连接后端服务";

function detailOf(payload: unknown): string | null {
  if (payload && typeof payload === "object" && "detail" in payload) {
    const detail = (payload as { detail: unknown }).detail;
    return typeof detail === "string" ? detail : null;
  }
  return null;
}

export async function request<T>(
  url: string,
  init?: RequestInit,
): Promise<ApiResult<T>> {
  let resp: Response;
  try {
    resp = await fetch(url, init);
  } catch {
    return { ok: false, detail: null, network: true };
  }
  const payloadPromise = resp.json().catch(() => null) as Promise<unknown>;
  if (!resp.ok) {
    return {
      ok: false,
      detail: detailOf(await payloadPromise),
      network: false,
    };
  }
  const data = await payloadPromise;
  if (data === null) {
    // 2xx 但响应体解析失败：与拆分前一样落到 catch，按网络故障处理。
    return { ok: false, detail: null, network: true };
  }
  return { ok: true, data: data as T };
}

const JSON_HEADERS = { "Content-Type": "application/json" };

// 带 JSON 请求体的写操作统一从这里取 headers，避免每处重复字面量。
export function jsonInit(method: string, body: unknown): RequestInit {
  return { method, headers: JSON_HEADERS, body: JSON.stringify(body) };
}

export function errorMessage(err: ApiErr, fallback: string): string {
  return err.network ? NETWORK_MESSAGE : err.detail ?? fallback;
}
