// In VM with Docker, Expo on host → FastAPI in container:
export const API_BASE_URL = "http://host.docker.internal:8012";

// For local dev without Docker, you can temporarily use:
// export const API_BASE_URL = "http://localhost:8012";
