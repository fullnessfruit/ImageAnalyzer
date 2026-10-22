/** A job may contain different URLs that return identical image bytes. Cache only successful inference. */
export class JobImageCache<T> {
  private readonly analyzed = new Map<string, { result: T; sourceUrl: string }>();

  async getOrAnalyze(
    imageSha256: string,
    imageUrl: string,
    analyze: () => Promise<T>,
  ): Promise<{ result: T; reusedFromImageUrl?: string }> {
    const previous = this.analyzed.get(imageSha256);
    if (previous) return { result: previous.result, reusedFromImageUrl: previous.sourceUrl };

    const result = await analyze();
    this.analyzed.set(imageSha256, { result, sourceUrl: imageUrl });
    return { result };
  }
}

/** Reuse a successful result for the same URL and analysis inputs across broker jobs. */
export class TimedImageResultCache<T> {
  private readonly entries = new Map<string, {
    result: T;
    imageSha256: string;
    bytes: number;
    expiresAt: number;
  }>();

  constructor(private readonly ttlMs = 24 * 60 * 60 * 1000, private readonly now = Date.now) {}

  get(url: string, analysisKey: string): { result: T; imageSha256: string; bytes: number } | undefined {
    const key = JSON.stringify([url, analysisKey]);
    const entry = this.entries.get(key);
    if (!entry) return undefined;
    if (entry.expiresAt <= this.now()) {
      this.entries.delete(key);
      return undefined;
    }
    return { result: entry.result, imageSha256: entry.imageSha256, bytes: entry.bytes };
  }

  set(url: string, analysisKey: string, result: T, imageSha256: string, bytes: number): void {
    this.entries.set(JSON.stringify([url, analysisKey]), {
      result, imageSha256, bytes, expiresAt: this.now() + this.ttlMs,
    });
  }

  prune(): void {
    const now = this.now();
    for (const [key, entry] of this.entries) {
      if (entry.expiresAt <= now) this.entries.delete(key);
    }
  }
}
