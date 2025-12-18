import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import ErrorBoundary from './components/utils/ErrorBoundary';
import { BrowserRouter, Route, Routes } from 'react-router-dom';

import CssBaseline from '@mui/material/CssBaseline';
import { Provider } from 'react-redux';
import { store } from './redux/store';
import AppAppBar from './components/ux/components/AppAppBar';

import Divider from '@mui/material/Divider';

//import FAQ from './components/ux/components/FAQ';
import Footer from './components/ux/components/Footer';
import AppTheme from './components/ux/shared-theme/AppTheme';

import Digitiser from './components/ux/components/FastDigitiser';
const rootElement = document.getElementById('root');
if (rootElement) {
  createRoot(rootElement).render(
    <StrictMode>
      <ErrorBoundary>
        <Provider store={store}>
          <BrowserRouter>
            <AppTheme>
              <CssBaseline enableColorScheme />
             
              <Routes>
                <Route
                  path="/"
                  element={
                    <>
                      <Digitiser />
                      <Divider />
                      <Footer />
                      <div style={{ display: 'none' }}>
                      </div>
                    </>
                  }
                />
              </Routes>
            </AppTheme>
          </BrowserRouter>
        </Provider>
      </ErrorBoundary>
    </StrictMode>,
  );
} else {
  console.error("Root element not found");
}