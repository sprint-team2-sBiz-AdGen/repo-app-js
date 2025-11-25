import { API_BASE_URL } from "./config";

async function handleResponse(res) {
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status} - ${text || res.statusText}`);
  }
  return res.json();
}

export async function translateDescription(descriptionKr) {
  const res = await fetch(`${API_BASE_URL}/translate-description`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ description_kr: descriptionKr }),
  });
  return handleResponse(res); // { id, description_kr, description_en }
}

export async function generateCopyVariants({
  descriptionId,
  strategyId,
  strategyName,
  productName,
  foregroundAnalysis = "",
}) {
  const res = await fetch(`${API_BASE_URL}/generate-copy-variants`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      description_id: descriptionId,
      strategy_id: strategyId,
      strategy_name: strategyName,
      product_name: productName,
      foreground_analysis: foregroundAnalysis,
    }),
  });
  return handleResponse(res); // { description_id, strategy_id, ..., variants: [{id, copy_ko}, ...] }
}

export async function uploadImage(imageUri) {
  const formData = new FormData();
  formData.append("file", {
    uri: imageUri,
    name: "upload.jpg",      // you can improve this later
    type: "image/jpeg",      // or infer from extension
  });

  const res = await fetch(`${API_BASE_URL}/upload-image`, {
    method: "POST",
    body: formData,
    // ⚠️ Do NOT set Content-Type manually; fetch will set the boundary for multipart/form-data
  });

  return handleResponse(res); // { id, original_filename }
}
