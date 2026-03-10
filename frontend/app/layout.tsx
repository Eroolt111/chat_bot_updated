import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'DBM Assistant',
  description: 'Development Bank of Mongolia AI Assistant',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link
          href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
