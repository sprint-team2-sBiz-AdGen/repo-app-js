import axios from 'axios';

const API_BASE_URL = 'http://34.9.178.28:8012';

async function handleResponse(res) {
  if (!res.ok) {
    const errorText = await res.text();
    console.error("API Error Response:", errorText);
    throw new Error(`Network request failed with status ${res.status}: ${errorText}`);
  }
  return res.json();
}

// export const translateDescription = async (description) => {
//   const response = await fetch(`${API_BASE_URL}/translate-description`, {
//     method: 'POST',
//     headers: {
//       'Content-Type': 'application/json',
//     },
//     body: JSON.stringify({ description_kr: description }),
//   });

//   if (!response.ok) {
//     throw new Error('Failed to translate description');
//   }
//   return response.json();
// };

// export const generateCopyVariants = async (payload) => {
//   const response = await fetch(`${API_BASE_URL}/generate-copy-variants`, {
//     method: 'POST',
//     headers: {
//       'Content-Type': 'application/json',
//     },
//     body: JSON.stringify(payload),
//   });

//   return handleResponse(response);
// };

export const uploadImage = async (imageUri) => {
  const formData = new FormData();

  const filename = imageUri.split('/').pop();

  const match = /\.(\w+)$/.exec(filename);
  const type = match ? `image/${match[1]}` : `image`;

  formData.append('file', {
    uri: imageUri,
    name: filename,
    type: type,
  });

  const response = await fetch(`${API_BASE_URL}/upload-image`, {
    method: 'POST',
    body: formData,
    headers: {

      'Content-Type': 'multipart/form-data',
    },
  });

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

export const createGenerationJob = async (imageUri, description) => {

  const formData = new FormData();
  const name = imageUri.split('/').pop() || 'upload.jpg';

  formData.append('image', {
    uri: imageUri,
    name: name,
    type: 'image/jpeg',
  });

  formData.append('request', JSON.stringify({ description }));

  const res = await fetch(`${API_BASE_URL}/api/v1/jobs/create`, {
    method: 'POST',
    body: formData,
  });

  const data = await res.json();
  if (!res.ok) throw new Error(JSON.stringify(data));
  return data;
};


export const gptKorToEng = async (jobId) => {
  const res = await fetch(`${API_BASE_URL}/api/js/gpt/kor-to-eng`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ job_id: jobId }),
  });

  const data = await res.json();
  if (!res.ok) throw new Error(JSON.stringify(data));
  return data;
};

export const gptAdCopyEng = async (jobId) => {
  const res = await fetch(`${API_BASE_URL}/api/js/gpt/ad-copy-eng`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ job_id: jobId }),
  });

  const data = await res.json();
  if (!res.ok) throw new Error(JSON.stringify(data));
  return data;
};

export const gptAdCopyKor = async (jobId) => {
  const res = await fetch(`${API_BASE_URL}/api/js/gpt/ad-copy-kor`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ job_id: jobId }),
  });

  const data = await res.json();
  if (!res.ok) throw new Error(JSON.stringify(data));
  return data;
};