import axios from 'axios';

// Use a single, consistent base URL
const API_BASE_URL = 'http://34.9.178.28:8012';

// Create a centralized axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 1 minute timeout
});

export const createGenerationJob = async (imageUri, description) => {
  const formData = new FormData();
  const name = imageUri.split('/').pop() || 'upload.jpg';

  formData.append('image', {
    uri: imageUri,
    name: name,
    type: 'image/jpeg',
  });

  // FIX 1: Use the correct form field name 'description'
  formData.append('description', description);

  try {
    // FIX 2: Use the correct endpoint '/jobs' and add Authorization header
    const response = await api.post('/jobs', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
        // This default token provides the tenant_id to the backend
        'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0ZW5hbnRfaWQiOiJkZWZhdWx0X3RlbmFudF9pZCJ9.5_V24T6a-3s_Gk3b_I-g_f_x_y_z'
      },
    });
    return response.data;
  } catch (error) {
    console.error("Error in createGenerationJob:", error.response ? error.response.data : error.message);
    throw error;
  }
};

export const gptKorToEng = async (jobId) => {
  try {
    const response = await api.post("/api/js/gpt/kor-to-eng", { job_id: jobId });
    return response.data;
  } catch (error) {
    console.error("Error in gptKorToEng:", error.response ? error.response.data : error.message);
    throw error;
  }
};

export const gptAdCopyEng = async (jobId) => {
  try {
    const response = await api.post("/api/js/gpt/ad-copy-eng", { job_id: jobId });
    return response.data;
  } catch (error) {
    console.error("Error in gptAdCopyEng:", error.response ? error.response.data : error.message);
    throw error;
  }
};

export const gptAdCopyKor = async (jobId) => {
  try {
    const response = await api.post("/api/js/gpt/ad-copy-kor", { job_id: jobId });
    return response.data;
  } catch (error) {
    console.error("Error in gptAdCopyKor:", error.response ? error.response.data : error.message);
    throw error;
  }
};

export const getJobResults = async (jobId) => {
  try {
    const response = await api.get(`/api/v1/jobs/${jobId}/results`);
    return response.data;
  } catch (error) {
    console.error(`Failed to fetch job results for ${jobId}:`, error.response ? error.response.data : error.message);
    throw error;
  }
};