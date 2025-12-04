import axios from "axios";
import { API_BASE_URL } from "../api/config";

// Create and export the axios instance as 'api'
export const api = axios.create({
  baseURL: API_BASE_URL || "http://34.9.178.28:8012", // Fallback for safety
  timeout: 2400000, // 1 minute timeout for requests
  headers: {
    "Content-Type": "application/json",
  },
});

// --- Existing Functions ---

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