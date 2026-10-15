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
