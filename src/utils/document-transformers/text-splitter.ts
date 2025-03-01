export class TextSplitter {
  split(text: string, maxLength: number = 1000): string[] {
    return text.match(new RegExp(`.{1,${maxLength}}`, 'g')) || [];
  }
}