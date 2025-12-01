-- Create table for storing community parlay datasets
CREATE TABLE public.parlay_datasets (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  filename TEXT NOT NULL,
  uploaded_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  data JSONB NOT NULL,
  upload_count INTEGER DEFAULT 0
);

-- Enable Row Level Security
ALTER TABLE public.parlay_datasets ENABLE ROW LEVEL SECURITY;

-- Create policy to allow anyone to view all datasets (public read)
CREATE POLICY "Anyone can view parlay datasets"
ON public.parlay_datasets
FOR SELECT
USING (true);

-- Create policy to allow anyone to upload datasets (public write)
CREATE POLICY "Anyone can upload parlay datasets"
ON public.parlay_datasets
FOR INSERT
WITH CHECK (true);

-- Create index for faster queries
CREATE INDEX idx_parlay_datasets_uploaded_at ON public.parlay_datasets(uploaded_at DESC);