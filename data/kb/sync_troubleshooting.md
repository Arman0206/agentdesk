# Sync Troubleshooting

## Error E-4102
Error E-4102 means the desktop client cannot reach the sync gateway. Check that port 443 is open on your firewall, then sign out and sign back in to refresh the device token.

## Error E-5310
Error E-5310 means a file exceeds the 5 GB per-file upload limit. Split the file or use the command line uploader, which supports resumable multipart uploads up to 50 GB.

## Slow sync
If sync is slow, pause and resume sync, exclude large build folders such as node_modules, and make sure bandwidth throttling is turned off in Settings under Network.
