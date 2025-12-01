-- Add UPDATE and DELETE policies for parlay_datasets
CREATE POLICY "Anyone can update parlay datasets"
ON public.parlay_datasets
FOR UPDATE
USING (true)
WITH CHECK (true);

CREATE POLICY "Anyone can delete parlay datasets"
ON public.parlay_datasets
FOR DELETE
USING (true);