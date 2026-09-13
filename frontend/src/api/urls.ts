// 后端产物音频的取用地址。

export function audioUrl(taskId: string, artifact: string): string {
  return `/api/tasks/${taskId}/audio/${
    artifact.split("/").map(encodeURIComponent).join("/")
  }`;
}
