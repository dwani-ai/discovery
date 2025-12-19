import { useState, useEffect, useRef } from 'react';

type ExtractionStatus = 'pending' | 'processing' | 'completed' | 'failed';

interface FileUploadResponse {
  file_id: string;
  filename: string;
  status: ExtractionStatus;
  message: string;
}

interface FileRetrieveResponse {
  file_id: string;
  filename: string;
  status: ExtractionStatus;
  extracted_text?: string;
  error_message?: string;
  created_at: string;
  updated_at: string;
}

export const useDocumentExtraction = () => {
  const [file, setFile] = useState<File | null>(null);
  const [fileId, setFileId] = useState<string | null>(null);
  const [extractedText, setExtractedText] = useState<string>('');
  const [status, setStatus] = useState<ExtractionStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const API_BASE = import.meta.env.VITE_DWANI_API_BASE_URL || 'https://discovery-server.dwani.ai';
  //const API_BASE = 'http://localhost:8000'
  const API_KEY = import.meta.env.VITE_DWANI_API_KEY;

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile && selectedFile.type === 'application/pdf') {
      setFile(selectedFile);
      setFileId(null);
      setExtractedText('');
      setStatus(null);
      setError(null);
      setPreviewUrl(null);
    } else {
      setError('Please select a valid PDF file.');
      setFile(null);
    }
  };

  const uploadFile = async () => {
    if (!file) return;

    setUploadLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE}/files/upload`, {
        method: 'POST',
        headers: {
          'accept': 'application/json',
          'X-API-KEY': API_KEY,
        },
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || 'Upload failed');
      }

      const data: FileUploadResponse = await response.json();
      setFileId(data.file_id);
      setStatus('pending');
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Failed to upload file'
      );
      setFileId(null);
    } finally {
      setUploadLoading(false);
    }
  };

  const fetchFileDetails = async (id: string) => {
    try {
      const response = await fetch(`${API_BASE}/files/${id}`, {
        headers: {
          'accept': 'application/json',
          'X-API-KEY': API_KEY,
        },
      });

      if (!response.ok) {
        throw new Error('Failed to fetch file details');
      }

      const data: FileRetrieveResponse = await response.json();
      setStatus(data.status);
      setExtractedText(data.extracted_text || '');
      if (data.status === 'failed') {
        setError(data.error_message || 'Extraction failed');
      }
      return data;
    } catch (err) {
      setError('Failed to fetch file details');
      throw err;
    }
  };

  // Poll for status updates
  useEffect(() => {
    if (!fileId || !status || ['completed', 'failed'].includes(status)) {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
      setLoading(false);
      return;
    }

    setLoading(true);
    fetchFileDetails(fileId);

    pollIntervalRef.current = setInterval(() => {
      fetchFileDetails(fileId);
    }, 3000);

    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, [fileId, status]);

  const loadExistingFile = async (id: string) => {
    setLoading(true);
    setError(null);
    setFile(null); // No file for existing
    setFileId(id);
    try {
      const data = await fetchFileDetails(id);
      setStatus(data.status);
      if (data.status === 'completed') {
        setExtractedText(data.extracted_text || '');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleStartExtraction = () => {
    if (file && !fileId) {
      uploadFile();
    }
  };

  const handleDownloadPdf = async () => {
    if (!fileId) return;

    try {
      const response = await fetch(`${API_BASE}/files/${fileId}/pdf`, {
        headers: {
          'X-API-KEY': API_KEY,
        },
      });

      if (!response.ok) {
        throw new Error('Failed to download PDF');
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `regenerated_${file?.name || 'document.pdf'}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      setError('Failed to download regenerated PDF');
    }
  };

  const handlePreviewPdf = async () => {
    if (!fileId || previewUrl) return;

    try {
      const response = await fetch(`${API_BASE}/files/${fileId}/pdf`, {
        headers: {
          'X-API-KEY': API_KEY,
        },
      });

      if (!response.ok) {
        throw new Error('Failed to fetch PDF');
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      setPreviewUrl(url);
    } catch (err) {
      setError('Failed to preview PDF');
    }
  };

  const reset = () => {
    setFile(null);
    setFileId(null);
    setExtractedText('');
    setStatus(null);
    setError(null);
    setLoading(false);
    setUploadLoading(false);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
    }
  };

  return {
    file,
    fileId,
    extractedText,
    status,
    loading: loading || uploadLoading,
    uploadLoading,
    error,
    previewUrl,
    handleFileChange,
    handleStartExtraction,
    handleDownloadPdf,
    handlePreviewPdf,
    loadExistingFile,
    reset,
    clearError: () => setError(null),
  };
};