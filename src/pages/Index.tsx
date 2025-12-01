import { useState, useEffect } from "react";
import { FileUploader } from "@/components/FileUploader";
import { StatsOverview } from "@/components/StatsOverview";
import { ParlayCard } from "@/components/ParlayCard";
import { Terminal } from "@/components/Terminal";
import { ThemeToggle } from "@/components/ThemeToggle";
import { ViewModeToggle, type ViewMode } from "@/components/ViewModeToggle";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Copy, Check, AlertCircle, BarChart3, Upload as UploadIcon, Edit, Trash2, Github } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { supabase } from "@/integrations/supabase/client";
import { cn } from "@/lib/utils";

interface ParlayData {
  generated_at: string;
  source_file: string;
  engine_config: any;
  num_parlays: number;
  parlays: any[];
}

interface Dataset {
  id: string;
  filename: string;
  uploadedAt: string;
  data: ParlayData;
}

const Index = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [currentDatasetId, setCurrentDatasetId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [terminalLogs, setTerminalLogs] = useState<string[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isMultiSelectMode, setIsMultiSelectMode] = useState(false);
  const [selectedParlays, setSelectedParlays] = useState<Set<number>>(new Set());
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [datasetToDelete, setDatasetToDelete] = useState<string | null>(null);
  const [renameDialogOpen, setRenameDialogOpen] = useState(false);
  const [datasetToRename, setDatasetToRename] = useState<Dataset | null>(null);
  const [newFilename, setNewFilename] = useState("");
  const [viewMode, setViewMode] = useState<ViewMode>("detailed");
  const { toast } = useToast();

  useEffect(() => {
    // Load all community datasets from database
    const loadDatasets = async () => {
      try {
        const { data: dbDatasets, error } = await supabase
          .from("parlay_datasets")
          .select("*")
          .order("uploaded_at", { ascending: false });

        if (error) throw error;

        if (dbDatasets && dbDatasets.length > 0) {
          const formattedDatasets: Dataset[] = dbDatasets.map((ds) => ({
            id: ds.id,
            filename: ds.filename,
            uploadedAt: ds.uploaded_at,
            data: ds.data as unknown as ParlayData,
          }));

          setDatasets(formattedDatasets);
          setCurrentDatasetId(formattedDatasets[0].id);
        } else {
          // If no datasets in DB, load sample data as fallback
          const res = await fetch("/sample-data.json");
          const data = await res.json();

          // Upload sample data to database
          const { data: inserted, error: insertError } = await supabase
            .from("parlay_datasets")
            .insert({
              filename: "sample-data.json",
              data: data,
            })
            .select()
            .single();

          if (!insertError && inserted) {
            const sampleDataset: Dataset = {
              id: inserted.id,
              filename: inserted.filename,
              uploadedAt: inserted.uploaded_at,
              data: inserted.data as unknown as ParlayData,
            };
            setDatasets([sampleDataset]);
            setCurrentDatasetId(sampleDataset.id);
          }
        }
      } catch (err) {
        console.error("Error loading datasets:", err);
        toast({
          title: "Error",
          description: "Failed to load community datasets",
          variant: "destructive",
        });
      } finally {
        setLoading(false);
      }
    };

    loadDatasets();
  }, [toast]);

  const currentDataset = datasets.find((d) => d.id === currentDatasetId);
  const parlayData = currentDataset?.data || null;

  const handleToggleMultiSelect = () => {
    setIsMultiSelectMode(!isMultiSelectMode);
    setSelectedParlays(new Set());
  };

  const handleToggleParlay = (index: number) => {
    const newSelected = new Set(selectedParlays);
    if (newSelected.has(index)) {
      newSelected.delete(index);
    } else {
      newSelected.add(index);
    }
    setSelectedParlays(newSelected);
  };

  const handleCopySelected = async () => {
    if (!parlayData || selectedParlays.size === 0) return;

    let text = `Selected Parlays (${selectedParlays.size})\n`;
    text += `Generated: ${new Date(parlayData.generated_at).toLocaleString()}\n`;
    text += "=".repeat(50) + "\n\n";

    Array.from(selectedParlays)
      .sort((a, b) => a - b)
      .forEach((index) => {
        const parlay = parlayData.parlays[index];
        text += `Parlay #${index + 1}\n`;
        text += `EV: ${(parlay.ev * 100).toFixed(2)}% | `;
        text += `Hit Prob: ${(parlay.hit_probability * 100).toFixed(2)}% | `;
        text += `Odds: ${parlay.implied_american}\n\n`;

        parlay.legs.forEach((leg: any, i: number) => {
          text += `Leg ${i + 1}: ${leg.player} (${leg.team})\n`;
          text += `  ${leg.proposition} @ ${leg.odds}\n`;
          text += `  Model: ${(leg.model_probability * 100).toFixed(1)}% | Market: ${(leg.implied_probability * 100).toFixed(1)}%\n\n`;
        });

        text += "=".repeat(50) + "\n\n";
      });

    await navigator.clipboard.writeText(text);
    toast({
      title: "Copied!",
      description: `${selectedParlays.size} parlays copied to clipboard`,
    });
  };

  const handleDeleteDataset = async () => {
    if (!datasetToDelete) return;

    try {
      const { error } = await supabase.from("parlay_datasets").delete().eq("id", datasetToDelete);

      if (error) throw error;

      // Remove from local state
      setDatasets((prev) => prev.filter((ds) => ds.id !== datasetToDelete));

      // If we deleted the current dataset, switch to the first available one
      if (currentDatasetId === datasetToDelete && datasets.length > 1) {
        const remainingDatasets = datasets.filter((ds) => ds.id !== datasetToDelete);
        setCurrentDatasetId(remainingDatasets[0]?.id || null);
      } else if (datasets.length === 1) {
        setCurrentDatasetId(null);
      }

      toast({
        title: "Deleted",
        description: "Dataset deleted successfully",
      });
    } catch (error) {
      console.error("Error deleting dataset:", error);
      toast({
        title: "Error",
        description: "Failed to delete dataset",
        variant: "destructive",
      });
    } finally {
      setDeleteDialogOpen(false);
      setDatasetToDelete(null);
    }
  };

  const handleRenameDataset = async () => {
    if (!datasetToRename || !newFilename.trim()) return;

    try {
      const { error } = await supabase
        .from("parlay_datasets")
        .update({ filename: newFilename.trim() })
        .eq("id", datasetToRename.id);

      if (error) throw error;

      // Update local state
      setDatasets((prev) =>
        prev.map((ds) => (ds.id === datasetToRename.id ? { ...ds, filename: newFilename.trim() } : ds)),
      );

      toast({
        title: "Renamed",
        description: "Dataset renamed successfully",
      });
    } catch (error) {
      console.error("Error renaming dataset:", error);
      toast({
        title: "Error",
        description: "Failed to rename dataset",
        variant: "destructive",
      });
    } finally {
      setRenameDialogOpen(false);
      setDatasetToRename(null);
      setNewFilename("");
    }
  };

  const openDeleteDialog = (datasetId: string) => {
    setDatasetToDelete(datasetId);
    setDeleteDialogOpen(true);
  };

  const openRenameDialog = (dataset: Dataset) => {
    setDatasetToRename(dataset);
    setNewFilename(dataset.filename);
    setRenameDialogOpen(true);
  };

  const handleFileSelect = async (file: File) => {
    setIsProcessing(true);
    setTerminalLogs([]);

    const addLog = (message: string) => {
      setTerminalLogs((prev) => [...prev, `[${new Date().toLocaleTimeString()}] ${message}`]);
    };

    addLog(`[INFO] Starting processing of ${file.name}`);
    addLog(`[INFO] File size: ${(file.size / 1024).toFixed(2)} KB`);

    try {
      // Check if it's a JSON file
      if (file.name.endsWith(".json")) {
        addLog("[INFO] Detected JSON file, loading directly...");

        const text = await file.text();
        const jsonData = JSON.parse(text);

        addLog("[SUCCESS] JSON file loaded successfully");
        addLog(`[INFO] Found ${jsonData.num_parlays} parlays in file`);
        addLog(`[INFO] Generated at ${new Date(jsonData.generated_at).toLocaleString()}`);
        addLog("[INFO] Uploading to community database...");

        // Save to database
        const { data: inserted, error: insertError } = await supabase
          .from("parlay_datasets")
          .insert({
            filename: file.name,
            data: jsonData,
          })
          .select()
          .single();

        if (insertError) {
          throw new Error("Failed to save to community database");
        }

        const newDataset: Dataset = {
          id: inserted.id,
          filename: inserted.filename,
          uploadedAt: inserted.uploaded_at,
          data: inserted.data as unknown as ParlayData,
        };

        setDatasets((prev) => [newDataset, ...prev]);
        setCurrentDatasetId(newDataset.id);

        addLog("[SUCCESS] Uploaded to community database");

        toast({
          title: "Success!",
          description: `Shared ${jsonData.num_parlays} parlays with the community`,
        });

        setIsProcessing(false);
        return;
      }

      // Process Excel file
      toast({
        title: "Processing...",
        description: `Analyzing ${file.name} and generating parlays...`,
      });

      const formData = new FormData();
      formData.append("file", file);

      addLog("[INFO] Uploading file to parlay engine...");

      const response = await fetch(`${import.meta.env.VITE_SUPABASE_URL}/functions/v1/process-parlays`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY}`,
        },
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        addLog(`[ERROR] Server responded with status ${response.status}`);
        addLog(`[ERROR] ${errorData.error || "Unknown error"}`);
        throw new Error(errorData.error || "Failed to process file");
      }

      addLog("[INFO] File uploaded successfully");
      addLog("[INFO] Running parlay engine algorithms...");
      addLog("[INFO] - Preprocessing props data");
      addLog("[INFO] - Estimating player minutes");
      addLog("[INFO] - Computing model probabilities");
      addLog("[INFO] - Building correlation matrix");
      addLog("[INFO] - Running Monte Carlo simulations");
      addLog("[INFO] - Searching for +EV parlays");

      const result = await response.json();

      addLog(`[SUCCESS] Generated ${result.num_parlays} parlay combinations`);
      addLog(`[INFO] Analysis complete at ${new Date(result.generated_at).toLocaleString()}`);
      addLog("[INFO] Uploading to community database...");

      // Save to database
      const { data: inserted, error: insertError } = await supabase
        .from("parlay_datasets")
        .insert({
          filename: file.name,
          data: result,
        })
        .select()
        .single();

      if (insertError) {
        throw new Error("Failed to save to community database");
      }

      const newDataset: Dataset = {
        id: inserted.id,
        filename: inserted.filename,
        uploadedAt: inserted.uploaded_at,
        data: inserted.data as unknown as ParlayData,
      };

      setDatasets((prev) => [newDataset, ...prev]);
      setCurrentDatasetId(newDataset.id);

      addLog("[SUCCESS] Shared with community");

      toast({
        title: "Success!",
        description: `Generated and shared ${result.num_parlays} parlay combinations`,
      });
    } catch (error: any) {
      console.error("Error processing file:", error);
      addLog(`[ERROR] Processing failed: ${error.message}`);
      toast({
        title: "Error",
        description: error.message || "Failed to process file. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsProcessing(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-background via-primary/5 to-accent/5">
        <div className="flex flex-col items-center gap-4">
          <div className="relative">
            <div className="w-16 h-16 rounded-full border-4 border-primary/30 border-t-primary animate-spin" />
            <div className="absolute inset-0 w-16 h-16 rounded-full bg-primary/10 animate-pulse-glow" />
          </div>
          <p className="text-lg font-bold text-gradient">Loading ParlayBud...</p>
        </div>
      </div>
    );
  }

  const stats = parlayData
    ? {
        num_parlays: parlayData.num_parlays,
        avgEV: parlayData.parlays.reduce((sum, p) => sum + p.ev, 0) / parlayData.parlays.length,
        avgHitProb: parlayData.parlays.reduce((sum, p) => sum + p.hit_probability, 0) / parlayData.parlays.length,
        maxLegs: Math.max(...parlayData.parlays.map((p) => p.num_legs)),
      }
    : { num_parlays: 0, avgEV: 0, avgHitProb: 0, maxLegs: 0 };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-primary/5 to-accent/5">
      {/* Header with frosted glass effect */}
      <header className="sticky top-0 z-50 border-b border-border/30">
        {/* Glass reflection layers with increased transparency */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-background/70 via-background/60 to-background/70 backdrop-blur-2xl" />
          <div className="absolute inset-0 bg-gradient-to-r from-primary/8 via-transparent to-accent/8" />
          <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-primary/40 to-transparent" />
          <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-border/60 to-transparent" />
          {/* Refraction effect */}
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_0%,rgba(59,130,246,0.08),transparent_50%)]" />
          {/* Subtle noise texture for glass effect */}
          <div
            className="absolute inset-0 opacity-[0.015] mix-blend-soft-light"
            style={{
              backgroundImage:
                "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' /%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)' /%3E%3C/svg%3E\")",
            }}
          />
        </div>

        <div className="container mx-auto px-6 py-6 relative">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="relative group">
                <div className="absolute inset-0 bg-gradient-to-br from-primary to-accent rounded-xl blur-md opacity-50 group-hover:opacity-75 transition-opacity" />
                <div className="relative p-3 rounded-xl bg-gradient-to-br from-primary to-accent shadow-lg">
                  <BarChart3 className="w-8 h-8 text-white" />
                </div>
              </div>
              <div>
                <h1 className="text-3xl font-bold text-foreground tracking-tight">ParlayBud</h1>
                <p className="text-sm text-muted-foreground font-medium mt-0.5">NBA Parlay Analysis</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <ThemeToggle />
              <div className="hidden sm:flex items-center gap-2 px-4 py-2 rounded-full bg-success/10 backdrop-blur-sm border border-success/20">
                <div className="w-2 h-2 rounded-full bg-success animate-pulse-glow" />
                <span className="text-xs font-semibold text-success">Live</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-6 py-8 lg:py-12">
        <Tabs defaultValue="parlays" className="w-full">
          <TabsList className="mb-8 p-1.5 bg-card/50 backdrop-blur-sm border border-border/50 shadow-lg">
            <TabsTrigger
              value="parlays"
              className="gap-2 data-[state=active]:bg-gradient-to-r data-[state=active]:from-primary data-[state=active]:to-accent data-[state=active]:text-white data-[state=active]:shadow-lg transition-all duration-300"
            >
              <BarChart3 className="w-4 h-4" />
              <span className="font-semibold">Parlays</span>
            </TabsTrigger>
            <TabsTrigger
              value="upload"
              className="gap-2 data-[state=active]:bg-gradient-to-r data-[state=active]:from-primary data-[state=active]:to-accent data-[state=active]:text-white data-[state=active]:shadow-lg transition-all duration-300"
            >
              <UploadIcon className="w-4 h-4" />
              <span className="font-semibold">Upload Data</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="parlays" className="space-y-8 animate-fade-in">
            {parlayData && (
              <>
                {/* Dataset Selector with enhanced styling */}
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h2 className="text-2xl font-bold text-foreground mb-1">Datasets</h2>
                      <p className="text-sm text-muted-foreground">
                        {datasets.length} shared dataset{datasets.length !== 1 ? "s" : ""} available
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-3">
                    <Select value={currentDatasetId || undefined} onValueChange={setCurrentDatasetId}>
                      <SelectTrigger className="flex-1 h-12 border-border/50 bg-card hover:border-primary/50 transition-all">
                        <SelectValue placeholder="Select a dataset" />
                      </SelectTrigger>
                      <SelectContent>
                        {datasets.map((dataset) => (
                          <SelectItem key={dataset.id} value={dataset.id}>
                            <div className="flex items-center gap-3">
                              <span className="font-semibold">{dataset.filename}</span>
                              <span className="text-xs text-muted-foreground">
                                {new Date(dataset.uploadedAt).toLocaleDateString()} •{" "}
                                {new Date(dataset.uploadedAt).toLocaleTimeString()}
                              </span>
                            </div>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>

                    {currentDataset && (
                      <div className="flex gap-2">
                        <Button
                          variant="outline"
                          size="icon"
                          onClick={() => openRenameDialog(currentDataset)}
                          title="Rename dataset"
                          className="h-12 w-12 hover:border-primary hover:text-primary transition-all"
                        >
                          <Edit className="h-5 w-5" />
                        </Button>
                        <Button
                          variant="destructive"
                          size="icon"
                          onClick={() => openDeleteDialog(currentDataset.id)}
                          title="Delete dataset"
                          className="h-12 w-12"
                        >
                          <Trash2 className="h-5 w-5" />
                        </Button>
                      </div>
                    )}
                  </div>
                </div>

                {/* Stats Overview */}
                <StatsOverview data={stats} />

                {/* Info Alert with gradient */}
                <Alert className="border-primary/30 bg-gradient-to-r from-primary/10 to-accent/10 backdrop-blur-sm shadow-lg animate-fade-in">
                  <AlertCircle className="h-5 w-5 text-primary" />
                  <AlertDescription className="text-sm font-medium">
                    Generated on{" "}
                    <span className="font-bold text-primary">{new Date(parlayData.generated_at).toLocaleString()}</span>{" "}
                    from <span className="font-bold text-foreground">{parlayData.source_file.split("/").pop()}</span>
                  </AlertDescription>
                </Alert>

                {/* Parlays Grid */}
                <div className="space-y-6">
                  <div className="flex items-center justify-between flex-wrap gap-4">
                    <div>
                      <h2 className="text-3xl font-bold text-gradient mb-1">Top Parlays by Expected Value</h2>
                      <p className="text-sm text-muted-foreground">Optimized combinations based on your data</p>
                    </div>

                    <div className="flex gap-3 items-center flex-wrap">
                      <ViewModeToggle viewMode={viewMode} onViewModeChange={setViewMode} />

                      {isMultiSelectMode && selectedParlays.size > 0 && (
                        <Button
                          onClick={handleCopySelected}
                          className="bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90 shadow-lg hover:shadow-xl transition-all duration-300"
                        >
                          <Copy className="w-4 h-4 mr-2" />
                          Copy Selected ({selectedParlays.size})
                        </Button>
                      )}
                      <Button
                        onClick={handleToggleMultiSelect}
                        variant={isMultiSelectMode ? "default" : "outline"}
                        className={cn(
                          "transition-all duration-300",
                          isMultiSelectMode &&
                            "bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90 shadow-lg",
                        )}
                      >
                        {isMultiSelectMode ? "Done" : "Multi-Select"}
                      </Button>
                    </div>
                  </div>

                  <div
                    className={cn(
                      "grid gap-6 transition-all duration-300",
                      viewMode === "detailed" && "grid-cols-1",
                      viewMode === "compact" && "grid-cols-1 md:grid-cols-2",
                      viewMode === "list" && "grid-cols-1 gap-3",
                    )}
                  >
                    {parlayData.parlays.map((parlay, index) => (
                      <ParlayCard
                        key={index}
                        parlay={parlay}
                        index={index}
                        viewMode={viewMode}
                        isSelectable={isMultiSelectMode}
                        isSelected={selectedParlays.has(index)}
                        onToggleSelect={() => handleToggleParlay(index)}
                      />
                    ))}
                  </div>
                </div>
              </>
            )}
          </TabsContent>

          <TabsContent value="upload" className="space-y-8 animate-fade-in">
            <div className="max-w-5xl mx-auto">
              <div className="mb-8 text-center">
                <h2 className="text-3xl font-bold text-gradient mb-3">Share Props Data with Community</h2>
                <p className="text-base text-muted-foreground max-w-2xl mx-auto">
                  Upload .xlsx files to generate AI-powered parlays, or share .json files directly • All uploads are
                  visible to the community
                </p>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <FileUploader onFileSelect={handleFileSelect} />
                <Terminal logs={terminalLogs} isProcessing={isProcessing} />
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </main>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Dataset</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete this dataset? This action cannot be undone and will remove it for all
              community members.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDeleteDataset}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Rename Dialog */}
      <Dialog open={renameDialogOpen} onOpenChange={setRenameDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Rename Dataset</DialogTitle>
            <DialogDescription>
              Enter a new name for this dataset. This will be visible to all community members.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <Label htmlFor="filename">Filename</Label>
              <Input
                id="filename"
                value={newFilename}
                onChange={(e) => setNewFilename(e.target.value)}
                placeholder="Enter new filename"
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    handleRenameDataset();
                  }
                }}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setRenameDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleRenameDataset} disabled={!newFilename.trim()}>
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Footer */}
      <footer className="relative border-t border-border/30 mt-16">
        {/* Glass effect for footer */}
        <div className="absolute inset-0 bg-gradient-to-br from-background/80 via-background/70 to-background/80 backdrop-blur-xl" />

        <div className="container mx-auto px-6 py-8 relative">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <span>© {new Date().getFullYear()} ParlayBud</span>
              <span className="hidden sm:inline">•</span>
              <span className="hidden sm:inline">© Vishal Thamaraimanalan</span>
            </div>

            <a
              href="https://github.com/yourusername/parlaybud"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-muted/50 hover:bg-muted transition-all duration-300 group border border-border/50 hover:border-primary/50"
            >
              <Github className="w-5 h-5 text-muted-foreground group-hover:text-primary transition-colors" />
              <span className="text-sm font-medium text-foreground">View on GitHub</span>
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Index;
