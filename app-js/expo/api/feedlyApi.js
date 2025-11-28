import { API_BASE_URL } from "./config";

async function handleResponse(res) {
  if (!res.ok) {
    // Try to get more specific error info from the server response
    const errorText = await res.text();
    console.error("API Error Response:", errorText);
    // Create a new error with a more descriptive message
    throw new Error(`Network request failed with status ${res.status}: ${errorText}`);
  }
  return res.json();
}

export const translateDescription = async (description) => {
  const response = await fetch(`${API_BASE_URL}/translate-description`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ description_kr: description }),
  });

  if (!response.ok) {
    throw new Error('Failed to translate description');
  }
  return response.json();
};

// --- FIX: Replace the entire generateCopyVariants function ---
export const generateCopyVariants = async (payload) => {
  // The 'payload' object contains { strategy_id, strategy_name, etc. }
  // We must stringify this object to send it as a JSON body.
  const response = await fetch(`${API_BASE_URL}/generate-copy-variants`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  return handleResponse(response);
};

export const uploadImage = async (imageUri) => {
  // Create a new FormData object, which is required for file uploads.
  const formData = new FormData();

  // The local image URI (e.g., 'file:///...') needs to be prepared for the network request.
  // We extract the filename from the end of the URI.
  const filename = imageUri.split('/').pop();
  
  // We infer the image's MIME type (e.g., 'image/jpeg') from the filename extension.
  const match = /\.(\w+)$/.exec(filename);
  const type = match ? `image/${match[1]}` : `image`;

  // This is the most critical step. We append the file data to the FormData object.
  // The key 'file' MUST match the argument name in your FastAPI endpoint:
  // `async def upload_image(file: UploadFile = File(...))`
  formData.append('file', {
    uri: imageUri,
    name: filename,
    type: type,
  });

  // We now send the request.
  const response = await fetch(`${API_BASE_URL}/upload-image`, {
    method: 'POST',
    body: formData,
    headers: {
      // This header is essential. It tells the server to expect a file, not JSON.
      'Content-Type': 'multipart/form-data',
    },
  });

  // The handleResponse function will process the server's reply.
  return handleResponse(response);
};

export const getLatestGenerations = async () => {
  const response = await fetch(`${API_BASE_URL}/generations`);

  if (!response.ok) {
    throw new Error("Failed to fetch latest generations");
  }
  return response.json();
};

export const getGenerationById = async (generationId) => {
  const response = await fetch(`${API_BASE_URL}/generations/${generationId}`);
  if (!response.ok) {
    throw new Error("Failed to fetch generation by ID");
  }
  return response.json();
};
