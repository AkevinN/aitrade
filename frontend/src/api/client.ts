/**
 * 全局 Axios 实例（baseURL、超时、拦截器）。
 *
 * - 请求拦截：对非 FormData 请求自动补充 `Content-Type: application/json`；
 *   FormData 请求不设 Content-Type，由浏览器自动附带 multipart boundary。
 * - 响应拦截：请求失败时统一提取 `detail` 或 `message`，写入 `console.error` 后透传异常。
 * - baseURL 优先读 `VITE_API_BASE_URL` 环境变量，缺省为 `http://localhost:8000`。
 */
import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
})

// Request interceptor: auto-set Content-Type only for non-FormData requests
api.interceptors.request.use(
  (config) => {
    if (config.data instanceof FormData) {
      // Let browser set Content-Type with boundary for FormData
      delete config.headers['Content-Type']
    } else if (!config.headers['Content-Type']) {
      config.headers['Content-Type'] = 'application/json'
    }
    return config
  },
  (error) => Promise.reject(error)
)

api.interceptors.response.use(
  (response) => response,
  (error) => {
    const msg = error.response?.data?.detail || error.message || '请求失败'
    console.error('[API Error]', msg)
    return Promise.reject(error)
  }
)

export default api
export { API_BASE_URL }
