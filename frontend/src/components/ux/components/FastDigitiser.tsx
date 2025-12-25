import { useState, useEffect } from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import Divider from '@mui/material/Divider';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';
import Chip from '@mui/material/Chip';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import DialogContentText from '@mui/material/DialogContentText';
import DialogTitle from '@mui/material/DialogTitle';
import TextField from '@mui/material/TextField';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemText from '@mui/material/ListItemText';
import ListItemAvatar from '@mui/material/ListItemAvatar';
import Avatar from '@mui/material/Avatar';
import Paper from '@mui/material/Paper';
import LinearProgress from '@mui/material/LinearProgress';
import SendIcon from '@mui/icons-material/Send';
import SearchIcon from '@mui/icons-material/Search';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Tooltip from '@mui/material/Tooltip';
import Checkbox from '@mui/material/Checkbox';
import Accordion from '@mui/material/Accordion';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import Drawer from '@mui/material/Drawer';
import IconButton from '@mui/material/IconButton';
import MenuIcon from '@mui/icons-material/Menu';
import DescriptionIcon from '@mui/icons-material/Description';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import DeleteIcon from '@mui/icons-material/Delete';
import LanguageIcon from '@mui/icons-material/Language';

import Highlight from 'react-highlight-words';

interface UploadedFile {
  file_id: string;
  filename: string;
  status: string;
  created_at: string;
}

interface LocalUpload {
  file: File;
  progress: number;
  status: 'uploading' | 'pending' | 'processing' | 'completed' | 'failed';
  file_id?: string;
  error?: string;
}

interface Source {
  filename: string;
  excerpt: string;
  relevance_score: number;
}

type Message = {
  role: 'user' | 'assistant' | 'system';
  content: string;
  sources?: Source[];
};

interface Conversation {
  id: string;
  fileIds: string[];
  filenames: string[];
  messages: Message[];
  lastUpdated: number;
  preview: string;
  isGlobal?: boolean;
}

const STORAGE_KEY = 'dwani_conversations';
const GLOBAL_CHAT_ID = 'global-all-documents';

export default function Digitiser() {
  const [chatOpen, setChatOpen] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [userMessage, setUserMessage] = useState('');
  const [chatHistory, setChatHistory] = useState<Message[]>([]);
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);

  const [searchQuery, setSearchQuery] = useState('');
  const [showSearchResults, setShowSearchResults] = useState(false);

  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [filesLoading, setFilesLoading] = useState(true);
  const [conversations, setConversations] = useState<Conversation[]>([]);

  const [localUploads, setLocalUploads] = useState<LocalUpload[]>([]);
  const [isUploading, setIsUploading] = useState(false);

  const [selectedFileIds, setSelectedFileIds] = useState<Set<string>>(new Set());
  const [activeChatFileIds, setActiveChatFileIds] = useState<string[]>([]);
  const [activeChatFilenames, setActiveChatFilenames] = useState<string[]>([]);

  // Delete confirmation
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [filesToDelete, setFilesToDelete] = useState<string[]>([]);

  const API_BASE = import.meta.env.VITE_DWANI_API_BASE_URL || 'https://discovery-server.dwani.ai';
  const API_KEY = import.meta.env.VITE_DWANI_API_KEY;

  // Load conversations from localStorage
  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      try {
        const parsed: Conversation[] = JSON.parse(saved);
        setConversations(parsed.sort((a, b) => b.lastUpdated - a.lastUpdated));
      } catch (e) {
        console.error('Failed to parse saved conversations');
      }
    }
  }, []);

  // Save conversations to localStorage
  useEffect(() => {
    if (conversations.length > 0) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
    }
  }, [conversations]);

  // Fetch uploaded files from server
  const fetchUploadedFiles = async () => {
    setFilesLoading(true);
    try {
      const response = await fetch(`${API_BASE}/files/`, {
        headers: { 'X-API-KEY': API_KEY || '' },
      });
      if (response.ok) {
        const data = await response.json();
        setUploadedFiles(data);
      }
    } catch (err) {
      console.error('Failed to fetch files list');
    } finally {
      setFilesLoading(false);
    }
  };

  useEffect(() => {
    fetchUploadedFiles();
    const interval = setInterval(fetchUploadedFiles, 5000);
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (s: string) => {
    switch (s) {
      case 'pending':
      case 'processing':
      case 'uploading':
        return 'warning';
      case 'completed':
        return 'success';
      case 'failed':
        return 'error';
      default:
        return 'default';
    }
  };

  const getStatusText = (s: string) => {
    switch (s) {
      case 'uploading': return 'Uploading';
      case 'pending': return 'Waiting';
      case 'processing': return 'Processing';
      case 'completed': return 'Ready';
      case 'failed': return 'Failed';
      default: return 'Unknown';
    }
  };

  const formatDate = (dateStr: string) => new Date(dateStr).toLocaleString();
  const formatRelativeTime = (timestamp: number) => {
    const now = Date.now();
    const diff = now - timestamp;
    const minutes = Math.floor(diff / 60000);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    if (days < 7) return `${days}d ago`;
    return new Date(timestamp).toLocaleDateString();
  };

  // Multiple file upload
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    const newUploads: LocalUpload[] = [];
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      if (file.type === 'application/pdf') {
        newUploads.push({
          file,
          progress: 0,
          status: 'uploading'
        });
      }
    }

    if (newUploads.length > 0) {
      setLocalUploads(prev => [...prev, ...newUploads]);
      uploadNextFile([...localUploads, ...newUploads]);
    }

    event.target.value = '';
  };

  const uploadNextFile = async (queue: LocalUpload[]) => {
    if (queue.length === 0 || isUploading) return;

    setIsUploading(true);
    const current = queue[0];
    const remaining = queue.slice(1);

    try {
      const formData = new FormData();
      formData.append('file', current.file);

      const response = await fetch(`${API_BASE}/files/upload`, {
        method: 'POST',
        headers: {
          'accept': 'application/json',
          'X-API-KEY': API_KEY || '',
        },
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || 'Upload failed');
      }

      const data = await response.json();
      setLocalUploads(prev => prev.map(u =>
        u === current
          ? { ...u, file_id: data.file_id, status: 'pending', progress: 100 }
          : u
      ));

    } catch (err) {
      setLocalUploads(prev => prev.map(u =>
        u === current
          ? { ...u, status: 'failed', error: err instanceof Error ? err.message : 'Upload failed' }
          : u
      ));
    } finally {
      setTimeout(() => {
        setLocalUploads(prev => prev.filter(u => u !== current));
        uploadNextFile(remaining);
        if (remaining.length === 0) setIsUploading(false);
      }, 1000);
    }
  };

  // Generate clean/merged PDF
  const handleGenerateMergedPdf = async () => {
    const selectedCompleted = allFiles
      .filter(f => selectedFileIds.has(f.file_id) && f.status === 'completed');

    if (selectedCompleted.length === 0) {
      alert('Please select at least one completed document.');
      return;
    }

    try {
      const response = await fetch(`${API_BASE}/files/merge-pdf`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-KEY': API_KEY || '',
        },
        body: JSON.stringify({
          file_ids: selectedCompleted.map(f => f.file_id)
        }),
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || 'Failed to generate PDF');
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;

      let downloadFilename;
      if (selectedCompleted.length === 1) {
        const originalName = selectedCompleted[0].filename.replace(/\.pdf$/i, '');
        downloadFilename = `clean_${originalName}.pdf`;
      } else {
        const baseNames = selectedCompleted
          .map(f => f.filename.replace(/\.pdf$/i, ''))
          .join('_');
        const dateStr = new Date().toISOString().slice(0, 10);
        downloadFilename = `merged_clean_${baseNames}_${dateStr}.pdf`;
      }

      a.download = downloadFilename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to generate PDF');
    }
  };

  // Delete files
  const handleDeleteFiles = async () => {
    if (filesToDelete.length === 0) return;

    try {
      const deletePromises = filesToDelete.map(fileId =>
        fetch(`${API_BASE}/files/${fileId}`, {
          method: 'DELETE',
          headers: {
            'X-API-KEY': API_KEY || '',
          },
        })
      );

      const results = await Promise.all(deletePromises);
      const failed = results.filter(r => !r.ok);

      if (failed.length > 0) {
        alert(`${failed.length} file(s) could not be deleted.`);
      }

      setUploadedFiles(prev => prev.filter(f => !filesToDelete.includes(f.file_id)));
      setSelectedFileIds(prev => {
        const newSet = new Set(prev);
        filesToDelete.forEach(id => newSet.delete(id));
        return newSet;
      });

      setConversations(prev => prev.filter(conv =>
        !conv.fileIds.some(id => filesToDelete.includes(id))
      ));

    } catch (err) {
      alert('Error deleting files. Please try again.');
    } finally {
      setDeleteDialogOpen(false);
      setFilesToDelete([]);
    }
  };

  const openDeleteDialog = (fileIds: string[]) => {
    setFilesToDelete(fileIds);
    setDeleteDialogOpen(true);
  };

  const toggleDrawer = () => setDrawerOpen(prev => !prev);
  const toggleSearch = () => {
    setShowSearchResults(prev => !prev);
    if (showSearchResults) setSearchQuery('');
  };

  const getConversationId = (fileIds: string[]) => fileIds.sort().join('|');

  const openChatForFiles = (fileIds: string[], filenames: string[], isGlobal = false) => {
    if (fileIds.length === 0) return;

    const convId = isGlobal ? GLOBAL_CHAT_ID : getConversationId(fileIds);
    const existing = conversations.find(c => c.id === convId);

    setActiveChatFileIds(fileIds);
    setActiveChatFilenames(filenames);
    setSelectedFileIds(new Set());
    setCurrentConversationId(convId);

    if (existing) {
      setChatHistory(existing.messages);
    } else {
      const welcomeMsg: Message = {
        role: 'assistant',
        content: isGlobal
          ? `**Global Chat** loaded with **${fileIds.length} completed document${fileIds.length > 1 ? 's' : ''}**.\n\nAsk anything — I'll search across your entire library!`
          : `I've loaded ${fileIds.length} document${fileIds.length > 1 ? 's' : ''}: ${filenames.join(', ')}.\n\nAsk me anything — I'll search across all of them!`
      };
      setChatHistory([welcomeMsg]);

      const newConv: Conversation = {
        id: convId,
        fileIds,
        filenames,
        messages: [welcomeMsg],
        lastUpdated: Date.now(),
        preview: welcomeMsg.content.slice(0, 60) + '...',
        isGlobal
      };
      setConversations(prev => [newConv, ...prev.filter(c => c.id !== convId)]);
    }
    setChatOpen(true);
  };

  const saveCurrentConversation = () => {
    if (!currentConversationId || activeChatFileIds.length === 0) return;

    const isGlobal = currentConversationId === GLOBAL_CHAT_ID;

    const currentFileIds = isGlobal
      ? allFiles.filter(f => f.status === 'completed').map(f => f.file_id)
      : activeChatFileIds;

    const currentFilenames = isGlobal
      ? allFiles.filter(f => f.status === 'completed').map(f => f.filename)
      : activeChatFilenames;

    setConversations(prev => {
      const filtered = prev.filter(c => c.id !== currentConversationId);
      const lastVisibleMsg = chatHistory.filter(m => m.role !== 'system').slice(-1)[0];
      const preview = lastVisibleMsg ? lastVisibleMsg.content.slice(0, 60) + '...' : 'New conversation';

      const updatedConv: Conversation = {
        id: currentConversationId,
        fileIds: currentFileIds,
        filenames: currentFilenames,
        messages: chatHistory,
        lastUpdated: Date.now(),
        preview,
        isGlobal
      };
      return [updatedConv, ...filtered];
    });
  };

  const handleCloseChat = () => {
    saveCurrentConversation();
    setChatOpen(false);
    setSearchQuery('');
    setShowSearchResults(false);
  };

  const handleSendMessage = async () => {
    if (!userMessage.trim() || chatLoading || activeChatFileIds.length === 0) return;

    const userMsg = userMessage.trim();
    setUserMessage('');
    setChatLoading(true);
    setChatError(null);

    const newUserMessage: Message = { role: 'user', content: userMsg };
    setChatHistory(prev => [...prev, newUserMessage]);

    const tempAssistantMessage: Message = { role: 'assistant', content: '', sources: [] };
    setChatHistory(prev => [...prev, tempAssistantMessage]);

    try {
      const res = await fetch(`${API_BASE}/chat-with-document`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-KEY': API_KEY || '',
        },
        body: JSON.stringify({
          file_ids: activeChatFileIds,
          messages: [
            ...chatHistory.filter(m => m.role !== 'system'),
            newUserMessage
          ]
        }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'Chat failed');
      }

      const data = await res.json();
      const answer = data.answer?.trim() || 'No reply.';
      const sources: Source[] = data.sources || [];

      setChatHistory(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = { role: 'assistant', content: answer, sources };
        return updated;
      });

      saveCurrentConversation();

    } catch (err) {
      setChatError(err instanceof Error ? err.message : 'Failed to get response');
      setChatHistory(prev => prev.slice(0, -2));
    } finally {
      setChatLoading(false);
    }
  };

  // === Combined files list with deduplication ===
  const serverFileIds = new Set(uploadedFiles.map(f => f.file_id));

  const serverDocs = uploadedFiles.map(f => ({
    ...f,
    source: 'server' as const,
    progress: 100,
    error: undefined,
  }));

  const localDocs = localUploads
    .filter(u => !u.file_id || !serverFileIds.has(u.file_id))
    .map((u, index) => ({
      file_id: u.file_id || `local-${u.file.name}-${u.file.size}-${u.file.lastModified}-${index}`,
      filename: u.file.name,
      status: u.status,
      created_at: new Date().toISOString(),
      source: 'local' as const,
      progress: u.progress,
      error: u.error,
    }));

  const allFiles = [...serverDocs, ...localDocs]
    .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());

  const completedCount = allFiles.filter(f => f.status === 'completed').length;
  const selectedCompletedCount = allFiles.filter(f => 
    selectedFileIds.has(f.file_id) && f.status === 'completed'
  ).length;

  return (
    <Box sx={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      p: { xs: 4, sm: 6 },
      pt: { xs: 10, sm: 12 },
      minHeight: '100vh',
      bgcolor: 'background.default',
    }}>
      <Stack direction="row" spacing={2} alignItems="center" sx={{ width: { xs: '100%', sm: '90%', md: '1000px' }, mb: 4 }}>
        <IconButton onClick={toggleDrawer} size="large" edge="start">
          <MenuIcon />
        </IconButton>
        <Typography variant="h5">dwani.ai – Document Intelligence</Typography>
      </Stack>

      <Stack spacing={5} sx={{ width: { xs: '100%', sm: '90%', md: '1000px' } }}>

        {/* Selection Action Bar */}
        {selectedFileIds.size > 0 && (
          <Alert
            severity="info"
            action={
              <Stack direction="row" spacing={1}>
                <Button
                  color="inherit"
                  size="small"
                  variant="outlined"
                  startIcon={<PictureAsPdfIcon />}
                  onClick={handleGenerateMergedPdf}
                  disabled={selectedCompletedCount === 0}
                >
                  Generate PDF ({selectedCompletedCount})
                </Button>
                <Button
                  color="inherit"
                  size="small"
                  variant="outlined"
                  startIcon={<DeleteIcon />}
                  onClick={() => openDeleteDialog(Array.from(selectedFileIds))}
                  disabled={selectedFileIds.size === 0}
                >
                  Delete ({selectedFileIds.size})
                </Button>
                <Button
                  color="inherit"
                  size="small"
                  variant="outlined"
                  onClick={() => {
                    const selected = allFiles
                      .filter(f => selectedFileIds.has(f.file_id) && f.status === 'completed');
                    if (selected.length > 0) {
                      openChatForFiles(
                        selected.map(f => f.file_id),
                        selected.map(f => f.filename)
                      );
                    }
                  }}
                  disabled={selectedCompletedCount === 0}
                >
                  Chat with {selectedFileIds.size}
                </Button>
              </Stack>
            }
            sx={{ mb: 2 }}
          >
            {selectedFileIds.size} document{selectedFileIds.size > 1 ? 's' : ''} selected
          </Alert>
        )}

        {/* Documents Header with Global Chat Button */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">
            Your Documents ({allFiles.length})
          </Typography>

          <Tooltip title={completedCount === 0 ? "Waiting for documents to finish processing" : "Chat with all completed documents"}>
            <span>
              <Button
                variant="contained"
                color="primary"
                startIcon={<LanguageIcon />}
                onClick={() => {
                  const completed = allFiles.filter(f => f.status === 'completed');
                  openChatForFiles(
                    completed.map(f => f.file_id),
                    completed.map(f => f.filename),
                    true
                  );
                }}
                disabled={completedCount === 0}
              >
                Global Chat ({completedCount})
              </Button>
            </span>
          </Tooltip>
        </Box>

        {/* Documents Table */}
        <Box>
          <TableContainer component={Paper} elevation={2} sx={{ maxHeight: 500, overflow: 'auto', borderRadius: 2 }}>
            <Table stickyHeader>
              <TableHead>
                <TableRow>
                  <TableCell padding="checkbox" sx={{ bgcolor: 'background.paper' }} />
                  <TableCell sx={{ bgcolor: 'background.paper', fontWeight: 'bold' }}>Filename</TableCell>
                  <TableCell sx={{ bgcolor: 'background.paper', fontWeight: 'bold' }}>Uploaded</TableCell>
                  <TableCell sx={{ bgcolor: 'background.paper', fontWeight: 'bold' }}>Status</TableCell>
                  <TableCell align="center" sx={{ bgcolor: 'background.paper', fontWeight: 'bold' }}>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {allFiles.length === 0 && !isUploading ? (
                  <TableRow>
                    <TableCell colSpan={5} align="center" sx={{ py: 8, color: 'text.secondary' }}>
                      No documents uploaded yet. Click below to add some!
                    </TableCell>
                  </TableRow>
                ) : (
                  allFiles.map((doc) => (
                    <TableRow key={doc.file_id} hover selected={selectedFileIds.has(doc.file_id)}>
                      <TableCell padding="checkbox">
                        <Checkbox
                          checked={selectedFileIds.has(doc.file_id)}
                          onChange={(e) => {
                            const newSet = new Set(selectedFileIds);
                            if (e.target.checked) newSet.add(doc.file_id);
                            else newSet.delete(doc.file_id);
                            setSelectedFileIds(newSet);
                          }}
                          disabled={doc.status !== 'completed'}
                        />
                      </TableCell>
                      <TableCell>
                        <Tooltip title={doc.filename}>
                          <Typography noWrap sx={{ maxWidth: 300 }}>
                            {doc.filename}
                          </Typography>
                        </Tooltip>
                      </TableCell>
                      <TableCell>{formatDate(doc.created_at)}</TableCell>
                      <TableCell>
                        <Stack spacing={1}>
                          <Chip
                            label={getStatusText(doc.status)}
                            color={getStatusColor(doc.status)}
                            size="small"
                          />
                          {doc.source === 'local' && doc.progress < 100 && (
                            <LinearProgress variant="determinate" value={doc.progress} />
                          )}
                          {doc.error && <Typography variant="caption" color="error">{doc.error}</Typography>}
                        </Stack>
                      </TableCell>
                      <TableCell align="center">
                        <Stack direction="row" spacing={1} justifyContent="center">
                          <Button
                            variant="outlined"
                            size="small"
                            onClick={() => openChatForFiles([doc.file_id], [doc.filename])}
                            disabled={doc.status !== 'completed'}
                          >
                            Chat
                          </Button>
                          {doc.source === 'server' && doc.status === 'completed' && (
                            <IconButton
                              size="small"
                              color="error"
                              onClick={() => openDeleteDialog([doc.file_id])}
                            >
                              <DeleteIcon fontSize="small" />
                            </IconButton>
                          )}
                        </Stack>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </Box>

        <Divider />

        {/* Upload Section */}
        <Typography variant="h6" gutterBottom>
          Upload Documents (Multiple PDFs Supported)
        </Typography>

        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} sx={{ alignItems: 'center' }}>
          <input
            type="file"
            accept="application/pdf"
            multiple
            onChange={handleFileChange}
            style={{ display: 'none' }}
            id="multi-pdf-upload"
            disabled={isUploading}
          />
          <label htmlFor="multi-pdf-upload">
            <Button
              variant="contained"
              component="span"
              startIcon={<UploadFileIcon />}
              disabled={isUploading}
            >
              {isUploading ? 'Uploading...' : 'Select PDFs'}
            </Button>
          </label>

          {localUploads.length > 0 && (
            <Typography variant="body2" color="text.secondary">
              {localUploads.filter(u => u.status === 'uploading').length} file(s) uploading...
            </Typography>
          )}
        </Stack>

        {localUploads.length > 0 && (
          <Alert severity="info" sx={{ mt: 2 }}>
            {localUploads.length} file(s) added. They will appear in the table as they process.
          </Alert>
        )}
      </Stack>

      {/* Conversations Drawer */}
      <Drawer anchor="left" open={drawerOpen} onClose={toggleDrawer}>
        <Box sx={{ width: 340, p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Conversations ({conversations.length})
          </Typography>
          <Divider sx={{ mb: 2 }} />
          <List>
            {/* Global Chat - Always at top if documents exist */}
            {completedCount > 0 && (
              <ListItem
                button
                onClick={() => {
                  const completed = allFiles.filter(f => f.status === 'completed');
                  openChatForFiles(
                    completed.map(f => f.file_id),
                    completed.map(f => f.filename),
                    true
                  );
                  setDrawerOpen(false);
                }}
                sx={{
                  borderRadius: 2,
                  mb: 2,
                  bgcolor: 'primary.main',
                  color: 'white',
                  '&:hover': { bgcolor: 'primary.dark' },
                }}
              >
                <ListItemAvatar>
                  <Avatar sx={{ bgcolor: 'white', color: 'primary.main' }}>
                    <LanguageIcon />
                  </Avatar>
                </ListItemAvatar>
                <ListItemText
                  primary="Global Chat"
                  secondary={`${completedCount} document${completedCount > 1 ? 's' : ''} • Search everything`}
                  primaryTypographyProps={{ fontWeight: 'bold' }}
                  secondaryTypographyProps={{ color: 'white', opacity: 0.9 }}
                />
              </ListItem>
            )}

            {/* Regular conversations */}
            {conversations.filter(c => !c.isGlobal).length === 0 && completedCount === 0 ? (
              <Typography color="text.secondary" sx={{ p: 4, textAlign: 'center' }}>
                No conversations yet.<br />Upload documents and start chatting!
              </Typography>
            ) : (
              conversations
                .filter(c => !c.isGlobal)
                .map(conv => (
                  <ListItem
                    button
                    key={conv.id}
                    onClick={() => {
                      openChatForFiles(conv.fileIds, conv.filenames);
                      setDrawerOpen(false);
                    }}
                    sx={{
                      borderRadius: 2,
                      mb: 1,
                      bgcolor: 'background.paper',
                      boxShadow: 1,
                      '&:hover': { boxShadow: 3 }
                    }}
                  >
                    <ListItemAvatar>
                      <Avatar sx={{ bgcolor: 'primary.main' }}>
                        <DescriptionIcon />
                      </Avatar>
                    </ListItemAvatar>
                    <ListItemText
                      primary={`${conv.filenames.length} document${conv.filenames.length > 1 ? 's' : ''}`}
                      secondary={
                        <>
                          <Typography component="span" variant="body2" color="text.primary" noWrap>
                            {conv.preview}
                          </Typography>
                          <br />
                          <Typography component="span" variant="caption" color="text.secondary">
                            {formatRelativeTime(conv.lastUpdated)}
                          </Typography>
                        </>
                      }
                    />
                  </ListItem>
                ))
            )}
          </List>
        </Box>
      </Drawer>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
        <DialogTitle>Confirm Deletion</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete {filesToDelete.length} document{filesToDelete.length > 1 ? 's' : ''}?
            This action cannot be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleDeleteFiles} color="error" variant="contained">
            Delete Permanently
          </Button>
        </DialogActions>
      </Dialog>

      {/* Chat Dialog */}
      <Dialog open={chatOpen} onClose={handleCloseChat} maxWidth="md" fullWidth>
        <DialogTitle>
          <Stack direction="row" justifyContent="space-between" alignItems="center">
            <Typography variant="h6">
              {currentConversationId === GLOBAL_CHAT_ID
                ? `Global Chat • ${activeChatFilenames.length} documents`
                : activeChatFilenames.length === 1
                ? activeChatFilenames[0]
                : `${activeChatFilenames.length} documents`}
            </Typography>
            <Button
              startIcon={<SearchIcon />}
              onClick={toggleSearch}
              variant={showSearchResults ? "contained" : "outlined"}
              size="small"
            >
              {showSearchResults ? 'Close Search' : 'Search'}
            </Button>
          </Stack>
        </DialogTitle>

        <DialogContent dividers sx={{ display: 'flex', flexDirection: 'column', height: '70vh' }}>
          {showSearchResults && activeChatFileIds.length === 1 && (
            <Paper variant="outlined" sx={{ p: 2, mb: 2, maxHeight: '40vh', overflow: 'auto', bgcolor: 'grey.50' }}>
              <Typography variant="subtitle1" gutterBottom>Search in document</Typography>
              <TextField
                fullWidth variant="outlined" placeholder="Search..."
                value={searchQuery} onChange={e => setSearchQuery(e.target.value)}
                sx={{ mb: 2 }} autoFocus
              />
              {searchQuery.trim() && (
                <Typography component="div" sx={{ whiteSpace: 'pre-wrap', fontSize: '0.95rem', lineHeight: 1.6 }}>
                  <Highlight
                    highlightClassName="search-highlight"
                    searchWords={[searchQuery.trim()]}
                    autoEscape
                    textToHighlight="Full document text search coming soon..."
                  />
                </Typography>
              )}
            </Paper>
          )}

          <Paper variant="outlined" sx={{ flexGrow: 1, overflow: 'auto', p: 2, mb: 2 }}>
            <List>
              {chatHistory.filter(m => m.role !== 'system').map((msg, i) => (
                <ListItem key={i} sx={{ justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start' }}>
                  <Paper elevation={1} sx={{
                    p: 2,
                    maxWidth: '85%',
                    bgcolor: msg.role === 'user' ? 'primary.light' : 'grey.100',
                    color: msg.role === 'user' ? 'white' : 'text.primary'
                  }}>
                    <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                      {msg.role === 'user' ? 'You' : 'Assistant'}
                    </Typography>
                    <Typography whiteSpace="pre-wrap" sx={{ mb: msg.sources && msg.sources.length > 0 ? 2 : 0 }}>
                      {msg.content || <i>Thinking...</i>}
                    </Typography>
                    {msg.sources && msg.sources.length > 0 && (
                      <Accordion elevation={0} sx={{ bgcolor: 'transparent', mt: 1 }}>
                        <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={{ fontSize: '0.875rem' }}>
                          <Typography variant="caption" fontWeight="medium">
                            Sources ({msg.sources.length})
                          </Typography>
                        </AccordionSummary>
                        <AccordionDetails sx={{ pt: 0 }}>
                          <Stack spacing={2}>
                            {msg.sources.map((source, idx) => (
                              <Box key={idx} sx={{ fontSize: '0.875rem' }}>
                                <Typography variant="caption" color="text.secondary" gutterBottom>
                                  <strong>{source.filename}</strong> (relevance: {source.relevance_score.toFixed(2)})
                                </Typography>
                                <Paper variant="outlined" sx={{ p: 1.5, bgcolor: 'grey.50', mt: 0.5 }}>
                                  <Highlight
                                    highlightClassName="source-highlight"
                                    searchWords={userMessage.split(' ').filter(w => w.length > 3)}
                                    textToHighlight={source.excerpt}
                                    autoEscape={true}
                                  />
                                </Paper>
                              </Box>
                            ))}
                          </Stack>
                        </AccordionDetails>
                      </Accordion>
                    )}
                  </Paper>
                </ListItem>
              ))}
              {chatLoading && (
                <ListItem>
                  <CircularProgress size={24} />
                  <Typography sx={{ ml: 2 }}>Searching documents and thinking...</Typography>
                </ListItem>
              )}
            </List>
          </Paper>

          {chatError && <Alert severity="error" onClose={() => setChatError(null)} sx={{ mb: 2 }}>{chatError}</Alert>}

          <Stack direction="row" spacing={1} alignItems="flex-end">
            <TextField
              fullWidth multiline maxRows={4} variant="outlined"
              placeholder="Ask a question about the document(s)..."
              value={userMessage} onChange={e => setUserMessage(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleSendMessage())}
              disabled={chatLoading}
            />
            <Button variant="contained" onClick={handleSendMessage} disabled={!userMessage.trim() || chatLoading} sx={{ height: 56 }}>
              <SendIcon />
            </Button>
          </Stack>
        </DialogContent>

        <DialogActions>
          <Button onClick={handleCloseChat}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}