# Implementation Plan: Smart Insect Identifier Platform

## Overview

This implementation plan breaks down the Smart Insect Identifier platform into logical, testable phases. The platform is a full-stack AI application with a FastAPI backend for ML inference and API services, and a Next.js frontend for the user interface. The implementation follows clean architecture principles with clear separation between ML services, external API integration (Google Gemini), and data persistence.

The plan is organized to deliver incremental value:
1. Core backend infrastructure and ML inference
2. API endpoints and service integration
3. Frontend foundation and UI components
4. Upload workflow and prediction display
5. History and additional features
6. Testing, polish, and deployment readiness

## Tasks

- [x] 1. Set up backend project structure and core infrastructure
  - Create FastAPI project with proper directory structure (routes/, services/, schemas/, utils/)
  - Set up Python virtual environment and install dependencies (FastAPI, PyTorch, Pillow, Pydantic, python-multipart)
  - Create configuration management system with environment variables
  - Implement structured logging with JSON output and configurable log levels
  - Create .env.example file documenting required environment variables
  - Set up CORS middleware for frontend integration
  - _Requirements: 16.7, 18.1, 18.2, 20.1, 20.5_

- [x] 2. Implement ML Service for model loading and inference
  - [x] 2.1 Create ML Service singleton class
    - Implement singleton pattern with private constructor and get_instance() method
    - Load PyTorch model from backend/artifacts/model.pth at initialization
    - Load class labels from backend/artifacts/labels.json
    - Load metadata (input size, normalization params) from backend/artifacts/metadata.json
    - Set model to evaluation mode and handle device selection (CPU/GPU)
    - Implement is_loaded() method for health checks
    - Add comprehensive error handling for missing files
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

  - [x] 2.2 Implement image preprocessing pipeline
    - Create preprocess_image() method accepting image bytes
    - Decode bytes to PIL Image and convert to RGB
    - Resize to model input dimensions from metadata
    - Apply normalization with mean and std from metadata
    - Convert to PyTorch tensor and add batch dimension
    - Validate image format and raise ValueError for invalid inputs
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [x] 2.3 Implement inference and prediction logic
    - Create predict() method accepting preprocessed tensor and top_k parameter
    - Run forward pass through model with torch.no_grad()
    - Apply softmax to get probability distributions
    - Extract top-k predictions with species names and confidence scores
    - Map class indices to species information from labels.json
    - Return PredictionResult with species, scientific name, confidence, and alternatives
    - _Requirements: 3.5, 3.6, 3.7_

  - [x] 2.4 Write unit tests for ML Service
    - Test singleton pattern ensures single instance
    - Test model loading validates file existence
    - Test image preprocessing with various formats
    - Test preprocessing rejects invalid image data
    - Test predict returns correct number of top-k predictions
    - Test confidence scores are between 0 and 1
    - Test tensor shapes match model requirements
    - _Requirements: 1.1, 1.2, 3.1, 3.7_

- [x] 3. Implement Gemini Service for AI insights
  - [x] 3.1 Create Gemini Service with retry logic
    - Implement GeminiService class with API key and timeout configuration
    - Create _build_prompt() method for structured insight requests
    - Implement async generate_insights() method with timeout wrapper
    - Add _request_with_retry() with exponential backoff (3 retries, 1s base delay)
    - Implement _is_transient_error() to classify retry-able errors (429, 500, 502, 503, 504)
    - Handle timeouts gracefully by returning None
    - Log all API requests, responses, and failures with timestamps
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 6.6, 6.7_

  - [x] 3.2 Design prompt for educational insights
    - Create structured prompt requesting taxonomy, physical characteristics, habitat, behavior, identification tips, and conservation status
    - Format prompt to return markdown-formatted content
    - Request sections with proper heading hierarchy
    - _Requirements: 5.2, 5.3_

  - [x] 3.3 Implement graceful degradation
    - Return None on permanent failures instead of raising exceptions
    - Return None on timeout without blocking
    - Return None on rate limiting (429) after retries
    - Log errors but don't propagate to caller
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [x] 3.4 Write unit tests for Gemini Service
    - Test generate_insights with successful API response
    - Test timeout handling returns None
    - Test retry logic with transient errors (503)
    - Test permanent errors (400) don't trigger retries
    - Test exponential backoff delays
    - Test _is_transient_error classification
    - Mock external API calls
    - _Requirements: 5.5, 6.1, 6.2, 6.3, 6.6_

- [x] 4. Implement History Service for data persistence
  - [x] 4.1 Create History Service with file-based storage
    - Implement HistoryService class with storage_path configuration
    - Create storage directory structure (storage/history.json, storage/images/)
    - Implement save_prediction() to store image, species, confidence, and timestamp
    - Generate unique IDs (UUID) for each record
    - Implement _save_image() to write image bytes to storage/images/
    - Return relative URL for saved images
    - _Requirements: 7.1, 7.4_

  - [x] 4.2 Implement history retrieval with filtering
    - Create get_history() method with limit, species_filter, date_from, date_to parameters
    - Load records from history.json
    - Apply case-insensitive species filtering
    - Apply date range filtering based on timestamps
    - Sort records by timestamp in descending order (newest first)
    - Return list of HistoryRecord objects
    - _Requirements: 7.2, 7.3, 7.6, 7.7_

  - [x] 4.3 Write unit tests for History Service
    - Test save_prediction creates record with correct fields
    - Test save_prediction stores image and returns URL
    - Test get_history filters by species name (case-insensitive)
    - Test get_history filters by date range
    - Test get_history sorts by timestamp descending
    - Test get_history respects limit parameter
    - Use temporary directory for test storage
    - _Requirements: 7.1, 7.2, 7.3, 7.6, 7.7_

- [x] 5. Define Pydantic schemas for API requests and responses
  - Create schemas/prediction.py with TopPrediction and PredictionResponse models
  - Create schemas/insights.py with InsightsRequest and InsightsResponse models
  - Create schemas/history.py with HistoryRecord and HistoryListResponse models
  - Create schemas/health.py with HealthResponse model
  - Create schemas/errors.py with ErrorResponse model
  - Add field validation constraints (confidence 0-1, non-empty strings, etc.)
  - Include JSON schema examples for OpenAPI documentation
  - _Requirements: 4.3, 5.7, 7.4, 8.4, 15.6_

- [x] 6. Implement API endpoints and dependency injection
  - [x] 6.1 Create prediction endpoint
    - Implement POST /api/predict route accepting file upload
    - Validate file format (PNG/JPG/JPEG/WEBP) and size (max 10MB)
    - Use dependency injection to get MLService and HistoryService instances
    - Read image bytes from uploaded file
    - Call MLService to preprocess and predict
    - Save prediction to history (non-blocking)
    - Return PredictionResponse with species, confidence, and top predictions
    - Handle errors: 400 for validation, 500 for inference failure, 503 for model not loaded
    - Respond within 3 seconds for images under 5MB
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.7, 14.4_

  - [x] 6.2 Create insights endpoint
    - Implement POST /api/insights route accepting species name
    - Validate species name is non-empty string
    - Use dependency injection to get GeminiService instance
    - Call GeminiService.generate_insights()
    - Always return 200 OK, even if insights unavailable (graceful degradation)
    - Return InsightsResponse with insights (or null), available flag, and error message
    - _Requirements: 5.6, 5.7, 6.1, 6.2, 6.3, 6.4_

  - [x] 6.3 Create history endpoint
    - Implement GET /api/history route with query parameters
    - Accept limit (1-500, default 100), species, date_from, date_to parameters
    - Validate query parameters with Pydantic
    - Use dependency injection to get HistoryService instance
    - Call HistoryService.get_history() with filters
    - Return list of HistoryRecord objects
    - _Requirements: 7.2, 7.3, 7.6, 7.7_

  - [x] 6.4 Create health check endpoint
    - Implement GET /api/health route
    - Check MLService.is_loaded() for model status
    - Check GeminiService availability (non-blocking)
    - Check HistoryService storage accessibility
    - Calculate uptime since server start
    - Return 200 if model loaded, 503 if critical services unavailable
    - Respond within 500 milliseconds
    - Return HealthResponse with all status fields
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_

  - [x] 6.5 Set up dependency injection and service lifecycle
    - Create dependency functions for service instances (get_ml_service, get_gemini_service, get_history_service)
    - Initialize MLService singleton on application startup
    - Initialize GeminiService with API key from environment
    - Initialize HistoryService with storage path from configuration
    - Register startup event to load model and verify artifacts
    - Add shutdown event to clean up resources if needed
    - _Requirements: 1.1, 16.6, 18.1_

  - [x] 6.6 Write integration tests for API endpoints
    - Test /api/predict with valid image returns 200 and prediction
    - Test /api/predict rejects invalid format with 400
    - Test /api/predict rejects oversized files with 413
    - Test /api/insights returns markdown content when available
    - Test /api/insights returns null gracefully when Gemini fails
    - Test /api/health reports model_loaded status correctly
    - Test /api/history returns records in reverse chronological order
    - Test /api/history filters by species and date range
    - Use FastAPI TestClient for endpoint testing
    - _Requirements: 4.1, 4.2, 4.4, 5.7, 6.4, 7.2, 7.3, 8.1, 8.2_

- [x] 7. Checkpoint - Backend verification
  - Ensure all backend tests pass (unit and integration)
  - Verify model loads successfully on startup
  - Test /api/predict endpoint with sample images manually
  - Test /api/health endpoint returns correct status
  - Verify Gemini API integration with test species
  - Confirm history storage creates files correctly
  - Ask the user if questions arise.

- [x] 8. Set up frontend Next.js project
  - Create Next.js project with TypeScript and App Router
  - Install dependencies (react, next, framer-motion, react-markdown, tailwindcss, shadcn/ui)
  - Configure Tailwind CSS with custom design tokens (colors, fonts, spacing)
  - Set up directory structure (app/, components/, services/, hooks/, types/, lib/)
  - Create layout.tsx with Inter font and dark theme base styles
  - Configure environment variables for API base URL
  - Set up ESLint and Prettier for code quality
  - _Requirements: 17.1, 17.2, 17.7, 18.5_

- [x] 9. Implement TypeScript types and API client
  - Create types/prediction.ts with PredictionResult, TopPrediction, PredictionError interfaces
  - Create types/insights.ts with InsightsResponse interface
  - Create types/history.ts with HistoryRecord, HistoryFilters interfaces
  - Create types/health.ts with HealthStatus interface
  - Create services/api.ts with APIClient class for HTTP requests
  - Implement APIClient.post() and APIClient.get() methods with error handling
  - Create services/prediction.ts with PredictionService class
  - Create services/insights.ts with InsightsService class (with graceful degradation)
  - Create services/history.ts with HistoryService class
  - Export singleton instances for each service
  - _Requirements: 17.3, 17.4, 6.4, 15.3_

- [x] 10. Build Shadcn UI component library
  - Install and configure Shadcn UI CLI
  - Add Button component with variants (primary, secondary, outline, ghost) and sizes (sm, md, lg)
  - Add Card component with title, hoverable states
  - Add Input component with validation states
  - Add DropZone component for drag-and-drop file uploads
  - Add Spinner and SkeletonLoader components for loading states
  - Add Toast notification component for error messages
  - Apply dark theme styling to all components
  - _Requirements: 17.6, 12.1, 12.2, 12.3, 14.5_

- [x] 11. Implement image upload components
  - [x] 11.1 Create ImageUploader component
    - Implement drag-and-drop functionality with file validation
    - Add file picker button as alternative upload method
    - Validate file format (PNG/JPG/JPEG/WEBP) client-side
    - Validate file size (max 10MB) client-side
    - Display error messages for invalid uploads
    - Show upload icon and instructional text
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.7_

  - [x] 11.2 Create ImagePreview component
    - Display uploaded image thumbnail
    - Show file name and size
    - Provide Remove button to clear image
    - Provide Replace button to upload different image
    - Provide Analyze button to trigger prediction
    - Apply card styling with dark theme
    - _Requirements: 2.5, 2.6_

  - [x] 11.3 Create AnalyzingLoader component
    - Display skeleton loaders during analysis
    - Show animated progress messages ("Analyzing image...", "Detecting species...")
    - Use Framer Motion for smooth animations
    - Apply gradient shimmer effect to skeleton elements
    - _Requirements: 9.3, 9.4, 14.5_

  - [x] 11.4 Write unit tests for upload components
    - Test ImageUploader validates file formats
    - Test ImageUploader validates file size limits
    - Test ImagePreview displays file information
    - Test Remove and Replace buttons trigger correct actions
    - Test AnalyzingLoader displays loading messages
    - Use React Testing Library
    - _Requirements: 2.3, 2.4, 2.7, 9.3_

- [x] 12. Implement prediction results components
  - [x] 12.1 Create SpeciesCard component
    - Display species name prominently with large font (text-4xl)
    - Display scientific name below in italics with secondary color
    - Display confidence score as percentage with color coding (green >80%, yellow 50-80%, orange <50%)
    - Add copy-to-clipboard button for species name
    - Apply card styling with dark background and border
    - Animate entrance with Framer Motion fade-in
    - _Requirements: 10.1, 10.2, 10.3, 10.7, 12.1, 12.2, 12.3_

  - [x] 12.2 Create TopPredictionsChart component
    - Display top 5 predictions with horizontal probability bars
    - Show species names and confidence percentages
    - Animate bar widths with Framer Motion (0 to percentage)
    - Sort predictions by confidence descending
    - Apply primary color gradient to bars
    - _Requirements: 10.4_

  - [x] 12.3 Create TaxonomyCard component
    - Display taxonomic classification (Kingdom, Phylum, Class, Order, Family, Genus, Species)
    - Show hierarchical structure with indentation or tree view
    - Use secondary text color for labels
    - Apply card styling consistent with SpeciesCard
    - _Requirements: 10.5_

  - [x] 12.4 Create PredictionResults container component
    - Compose SpeciesCard, TopPredictionsChart, and TaxonomyCard
    - Display prediction timestamp
    - Apply spacing between sections (spacing-6)
    - Wrap in Framer Motion container for staggered animations
    - _Requirements: 10.1, 10.4, 10.5, 10.6_

  - [x] 12.5 Write unit tests for prediction components
    - Test SpeciesCard displays all information correctly
    - Test confidence color coding (green/yellow/orange)
    - Test TopPredictionsChart renders correct number of bars
    - Test TopPredictionsChart sorts by confidence
    - Test TaxonomyCard displays taxonomic hierarchy
    - Test copy-to-clipboard functionality
    - _Requirements: 10.1, 10.3, 10.4, 10.7_

- [x] 13. Implement AI insights components
  - [x] 13.1 Create AIInsightsSection component
    - Integrate react-markdown for rendering markdown content
    - Create custom component mappings for headings, paragraphs, lists, code blocks
    - Apply prose styling with proper typography (line height, spacing)
    - Display InsightsLoader while fetching from API
    - Show UnavailableMessage when Gemini API fails (graceful degradation)
    - Ensure markdown renders without breaking on malformed content
    - Apply dark theme styling to rendered content
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7_

  - [ ] 13.2 Create useInsights custom hook
    - Accept species name as parameter
    - Manage loading, data, and error states
    - Call InsightsService.getInsights() on species change
    - Return data, isLoading, error for component consumption
    - Handle null response gracefully (Gemini unavailable)
    - _Requirements: 17.2, 5.7, 6.4_

  - [ ] 13.3 Style markdown content with prose classes
    - Configure react-markdown with custom renderers for headings, paragraphs, lists, tables, quotes, code blocks
    - Apply consistent typography with Inter font
    - Use appropriate text sizes for heading hierarchy
    - Add spacing between sections
    - Style code blocks with monospace font and background
    - Ensure WCAG AA contrast ratios
    - _Requirements: 11.2, 11.3, 11.7, 12.7, 13.4_

  - [ ] 13.4 Write unit tests for insights components
    - Test AIInsightsSection displays loading state
    - Test AIInsightsSection renders markdown correctly
    - Test AIInsightsSection shows unavailable message on error
    - Test useInsights hook fetches data on species change
    - Test useInsights handles null response gracefully
    - Mock InsightsService for predictable testing
    - _Requirements: 11.4, 11.5, 6.4_

- [ ] 14. Implement upload workflow state machine
  - Create useUploadWorkflow custom hook
  - Define WorkflowStep type ('upload' | 'preview' | 'analyzing' | 'results' | 'insights')
  - Implement state management for currentStep, selectedImage, prediction, insights, error
  - Create selectImage() function to transition from upload to preview
  - Create analyze() async function to call PredictionService and transition to results
  - Create fetchInsights() async function to call InsightsService (non-blocking)
  - Handle errors gracefully and transition back to appropriate step
  - Implement reset() function to restart workflow
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 17.2_

- [ ] 15. Build analyze page with upload workflow
  - Create app/analyze/page.tsx
  - Integrate useUploadWorkflow hook
  - Conditionally render components based on currentStep
  - Wire ImageUploader onImageSelected to selectImage()
  - Wire ImagePreview analyze button to analyze()
  - Display AnalyzingLoader during analyzing step
  - Display PredictionResults during results step
  - Display AIInsightsSection during insights step
  - Add Framer Motion page transitions
  - Apply responsive layout with proper spacing
  - _Requirements: 9.1, 9.2, 9.7, 10.1, 11.1, 13.4, 17.5_

- [ ] 16. Implement history page with filtering
  - [ ] 16.1 Create HistoryCard component
    - Display thumbnail image (lazy-loaded)
    - Display species name and scientific name
    - Display confidence score with color coding
    - Display timestamp in human-readable format
    - Make card clickable to view full prediction details
    - Apply card styling with hover effects
    - _Requirements: 7.5_

  - [ ] 16.2 Create HistoryGrid component
    - Render grid of HistoryCard components
    - Implement responsive grid (1-4 columns based on viewport)
    - Implement lazy loading for images (only load visible cards)
    - Handle empty state when no records exist
    - _Requirements: 7.5, 14.2, 13.4_

  - [ ] 16.3 Create FilterPanel component
    - Add search input for species name filtering
    - Add date range picker for temporal filtering
    - Apply filters on input change (client-side filtering)
    - Display active filters count
    - Provide clear filters button
    - _Requirements: 7.6, 7.7_

  - [ ] 16.4 Create useHistory custom hook
    - Accept filters as parameter
    - Fetch history from HistoryService on mount
    - Implement refresh() function to reload data
    - Manage loading, data, and error states
    - _Requirements: 17.2, 7.2, 7.3_

  - [ ] 16.5 Build history page
    - Create app/history/page.tsx
    - Integrate useHistory hook
    - Render FilterPanel and HistoryGrid
    - Apply page layout with header and spacing
    - Add error handling and empty states
    - _Requirements: 7.5, 13.4_

  - [ ] 16.6 Write unit tests for history components
    - Test HistoryCard displays all fields correctly
    - Test HistoryGrid renders responsive layout
    - Test FilterPanel applies filters correctly
    - Test useHistory hook fetches and filters data
    - Test lazy loading only loads visible images
    - _Requirements: 7.5, 7.6, 7.7_

- [ ] 17. Implement error handling and user feedback
  - [ ] 17.1 Create ErrorBoundary component
    - Implement React error boundary with getDerivedStateFromError
    - Create ErrorFallback component with retry button
    - Log errors to console (prepare for error tracking service integration)
    - Wrap app in ErrorBoundary for global error catching
    - _Requirements: 15.1_

  - [ ] 17.2 Implement API error handling
    - Create handleAPIError utility function
    - Map HTTP status codes to user-friendly messages
    - Handle network errors separately
    - Return actionable error messages
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5_

  - [ ] 17.3 Add Toast notifications
    - Integrate toast library (react-hot-toast or similar)
    - Display toast on upload errors
    - Display toast on API failures
    - Display toast on history save failures (non-blocking)
    - Style toasts with dark theme
    - _Requirements: 15.1, 15.4_

  - [ ] 17.4 Create error message components
    - Create ErrorMessage component for inline errors
    - Create UnavailableMessage component for graceful degradation
    - Apply consistent styling across error states
    - Include actionable suggestions in messages
    - _Requirements: 15.1, 15.5, 6.4_

  - [ ] 17.5 Write tests for error handling
    - Test ErrorBoundary catches rendering errors
    - Test handleAPIError maps status codes correctly
    - Test Toast displays on API failures
    - Test UnavailableMessage shows when Gemini fails
    - _Requirements: 15.1, 15.2, 15.6, 6.4_

- [ ] 18. Implement accessibility features
  - Add ARIA labels to all interactive elements (buttons, inputs, links)
  - Ensure keyboard navigation works for upload workflow
  - Provide visible focus indicators (2px outline) for all focusable elements
  - Add alt text for all images and icons
  - Ensure minimum 44x44px touch targets for mobile
  - Test color contrast meets WCAG AA standards (use contrast checker)
  - Support browser zoom up to 200% without breaking layout
  - Add skip-to-content link for keyboard users
  - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7_

- [ ] 19. Implement responsive design
  - Test layout on mobile (320px), tablet (768px), and desktop (1920px+) viewports
  - Ensure ImageUploader works on touch devices
  - Make HistoryGrid responsive (1 column mobile, 2-3 tablet, 4 desktop)
  - Adjust font sizes for readability on small screens
  - Test upload workflow on various devices
  - Ensure all interactive elements are touch-friendly (min 44x44px)
  - Use Tailwind responsive utilities (sm:, md:, lg:, xl:)
  - _Requirements: 13.4, 13.6_

- [ ] 20. Optimize performance
  - [ ] 20.1 Implement frontend performance optimizations
    - Add dynamic imports for route-based code splitting
    - Implement lazy loading for HistoryCard images
    - Add skeleton loaders for perceived performance
    - Optimize images before upload (client-side compression if needed)
    - Add loading states within 100ms of user actions
    - _Requirements: 14.1, 14.2, 14.3, 14.5, 14.6_

  - [ ] 20.2 Configure caching and optimization headers
    - Set cache headers for static assets in Next.js
    - Configure image optimization in next.config.js
    - Enable compression for API responses (gzip)
    - _Requirements: 14.7_

  - [ ] 20.3 Run performance audits
    - Run Lighthouse audit and aim for >90 performance score
    - Measure First Contentful Paint (<1.5s on 3G)
    - Test prediction API response time (<3s for 5MB images)
    - Verify code splitting reduces initial bundle size
    - _Requirements: 14.1, 14.4_

- [ ] 21. Create home page and navigation
  - Create app/page.tsx with hero section
  - Add headline, subtitle, and CTA button to start analyzing
  - Create features section highlighting key capabilities
  - Add navigation header with logo and links (Analyze, History)
  - Style with dark theme and animations
  - Link CTA button to /analyze page
  - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

- [ ] 22. Add structured logging to backend
  - Configure logging format as JSON for log aggregation
  - Log all incoming API requests with endpoint, method, timestamp
  - Log ML inference duration and prediction results
  - Log Gemini API requests, responses, and errors
  - Log errors with stack traces and contextual information
  - Ensure no sensitive data (API keys, images) are logged
  - Make log level configurable via environment variable (DEBUG, INFO, WARNING, ERROR)
  - _Requirements: 20.1, 20.2, 20.3, 20.4, 20.5, 20.6, 20.7_

- [ ] 23. Configure environment variables and deployment
  - Document all required environment variables in .env.example (backend)
  - Document GEMINI_API_KEY, API_HOST, API_PORT, STORAGE_PATH, LOG_LEVEL
  - Create .env.local.example for frontend with NEXT_PUBLIC_API_URL
  - Add .env files to .gitignore to prevent secret leakage
  - Validate required environment variables on backend startup
  - Create README.md with setup instructions
  - _Requirements: 18.1, 18.2, 18.3, 18.4, 18.5, 18.6, 18.7_

- [ ] 24. Set up OpenAPI documentation
  - Verify FastAPI auto-generates OpenAPI schema at /docs
  - Add detailed docstrings to all API endpoints
  - Document request and response schemas in route functions
  - Include example requests and responses in schemas
  - Document error codes and their meanings
  - Add tags to group related endpoints
  - Test /docs endpoint renders correctly
  - _Requirements: 19.1, 19.2, 19.3, 19.4, 19.6, 19.7_

- [ ] 25. Checkpoint - Full system integration test
  - Start backend server and verify model loads
  - Open frontend in browser and navigate through all pages
  - Test complete upload workflow: select image → preview → analyze → view results → view insights
  - Test graceful degradation: temporarily disable Gemini API and verify predictions still work
  - Test history: verify predictions are saved and can be retrieved with filters
  - Test error handling: upload invalid file and verify error message
  - Test responsive design on mobile, tablet, and desktop viewports
  - Test accessibility: navigate with keyboard only
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 26. Polish and final touches
  - [ ] 26.1 Refine animations and transitions
    - Fine-tune Framer Motion timing and easing
    - Add subtle hover effects to interactive elements
    - Ensure animations respect prefers-reduced-motion
    - Test animations on various devices
    - _Requirements: 9.7_

  - [ ] 26.2 Improve loading states
    - Add skeleton loaders for all async operations
    - Ensure loading indicators appear within 100ms
    - Add progress messages for long operations
    - Test loading states with slow network simulation
    - _Requirements: 14.5_

  - [ ] 26.3 Add final error handling edge cases
    - Handle model loading failure on startup
    - Handle storage permission errors
    - Handle network disconnection during upload
    - Add retry mechanisms for transient failures
    - _Requirements: 15.1, 15.2, 15.3_

  - [ ] 26.4 Optimize dark theme styling
    - Verify all colors meet WCAG AA contrast ratios
    - Ensure consistent spacing and typography
    - Polish card shadows and borders
    - Test theme on various monitors and devices
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7_

  - [ ] 26.5 Run final quality checks
    - Run all test suites (backend unit, integration, frontend unit tests)
    - Run Lighthouse audit for performance, accessibility, SEO
    - Test on Chrome, Firefox, Safari browsers
    - Verify OpenAPI documentation is complete
    - Check for console errors or warnings
    - _Requirements: 14.1, 13.1, 19.1_

- [ ] 27. Create production deployment artifacts
  - Build frontend for production (npm run build)
  - Create Docker configuration for backend (Dockerfile, docker-compose.yml)
  - Document deployment steps in README.md
  - Create systemd service file for backend (if deploying to Linux server)
  - Set up environment variable templates for production
  - Document system requirements (Python 3.10+, Node 18+, disk space for model)
  - _Requirements: 18.1, 18.4_

- [ ] 28. Final checkpoint and handoff
  - Verify all 20 requirements are implemented
  - Ensure all tests pass
  - Verify deployment documentation is complete
  - Test end-to-end workflow one final time
  - Create user guide or demo video (optional)
  - Ask the user if questions arise or if any adjustments are needed.

## Notes

- Tasks marked with `*` are optional test-related sub-tasks that can be skipped for faster MVP delivery
- Each task references specific requirement numbers for traceability
- The implementation follows clean architecture with clear service boundaries
- Graceful degradation is built into the design (predictions work even if Gemini fails)
- The plan emphasizes incremental testing with checkpoints at major milestones
- Frontend and backend can be developed in parallel after Phase 1 completion
- All components use TypeScript for type safety
- All services use async/await for non-blocking operations
- The design system ensures consistent styling across all components

## Task Dependency Graph

```json
{
  "waves": [
    {
      "id": 0,
      "tasks": ["1", "8"]
    },
    {
      "id": 1,
      "tasks": ["2.1", "3.1", "4.1", "9", "10"]
    },
    {
      "id": 2,
      "tasks": ["2.2", "3.2", "4.2", "5", "11.1"]
    },
    {
      "id": 3,
      "tasks": ["2.3", "3.3", "11.2", "11.3"]
    },
    {
      "id": 4,
      "tasks": ["2.4", "3.4", "4.3", "11.4"]
    },
    {
      "id": 5,
      "tasks": ["6.1", "6.2", "6.3", "6.4", "12.1", "12.2", "12.3"]
    },
    {
      "id": 6,
      "tasks": ["6.5", "12.4", "12.5", "13.1", "13.2"]
    },
    {
      "id": 7,
      "tasks": ["6.6", "13.3", "13.4", "14"]
    },
    {
      "id": 8,
      "tasks": ["15", "16.1", "16.2", "16.3", "16.4"]
    },
    {
      "id": 9,
      "tasks": ["16.5", "16.6", "17.1", "17.2", "17.3", "17.4"]
    },
    {
      "id": 10,
      "tasks": ["17.5", "18", "19", "20.1", "20.2"]
    },
    {
      "id": 11,
      "tasks": ["20.3", "21", "22", "23", "24"]
    },
    {
      "id": 12,
      "tasks": ["26.1", "26.2", "26.3", "26.4"]
    },
    {
      "id": 13,
      "tasks": ["26.5", "27"]
    }
  ]
}
```
