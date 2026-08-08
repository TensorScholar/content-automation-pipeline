# Smarlux P0 Live Canary

These checks require real external systems and therefore cannot be proven by a
source-only package. Run them against a disposable WordPress staging site and a
read-only Google Search Console property before public promotion. Record every
request, task, attempt, remote ID, sync-run ID, timestamp, and result against the
immutable release commit.

## WordPress canary

### Connection and permission qualification

1. Configure a dedicated WordPress Application Password over HTTPS.
2. Verify the account can edit posts.
3. Verify draft publication succeeds.
4. Verify scheduled/public publication remains blocked when `publish_posts` is
   absent.
5. Verify invalid and revoked credentials fail without retrying indefinitely.

### Draft, idempotency, and update

1. Publish one approved article as `draft`.
2. Confirm local success only after the remote post can be read back with the
   expected ID, slug, status, and URL.
3. Submit the identical request again.
4. Confirm the same WordPress post is returned and no duplicate post exists.
5. Edit the article content and publish again.
6. Confirm the existing remote post is updated rather than duplicated.

### Optional metadata degradation

1. Use a WordPress site without one optional SEO/meta capability.
2. Confirm the core post still succeeds when safe to do so.
3. Confirm the unsupported optional operation appears as a structured warning.
4. Confirm Smarlux does not claim that schema or SEO metadata was stored when it
   was not.

### Failure and recovery

1. Cause a temporary 5xx or network interruption before remote creation and
   confirm bounded retry behavior.
2. Interrupt the client response after WordPress creates the post.
3. Confirm reconciliation finds the post by remote ID or deterministic slug and
   repairs local state without creating a duplicate.
4. Restart the Celery worker during a leased attempt and confirm stale-attempt
   recovery.
5. Confirm a late failure/retry from an older competing task cannot regress an
   already successful article.

### Public publication

1. Confirm a non-manager cannot request scheduled/public publication.
2. Confirm a manager request remains subject to article approval and risk gates.
3. Publish exactly one approved article publicly.
4. Verify remote state and URL.
5. Repeat the same request and confirm no duplicate.

## Search Console canary

### OAuth security

1. Start the connection from the intended project.
2. Confirm the authorization request asks only for
   `https://www.googleapis.com/auth/webmasters.readonly`.
3. Confirm a modified/replayed state is rejected.
4. Complete consent and confirm the refresh token is stored encrypted.
5. Confirm the browser returns to the configured HTTPS frontend URL.

### Property access and sync

1. Refresh accessible properties.
2. Select one URL-prefix or domain property the account can read.
3. Run one completed reporting window.
4. Confirm page-level clicks, impressions, CTR, and average position are stored.
5. Run the same project/property/window again and confirm the sync is idempotent.
6. Confirm pagination and any `truncated` result are visible in sync status.
7. Confirm manual CSV import still works as a fallback and remains a distinct
   source.

### Failure and recovery

1. Revoke the OAuth grant and confirm the next sync becomes a durable revoked
   credential error rather than looping forever.
2. Restore access by reconnecting.
3. Simulate a temporary quota/5xx response and confirm bounded retry.
4. Restart the worker during a sync and confirm stale-run reconciliation.
5. Remove property access and confirm the selected property is cleared and sync
   fails closed.
6. Disconnect and confirm the encrypted refresh token is removed while audit
   history remains.

## Backup/restore canary

Run:

```bash
scripts/maintenance/verify_backup_restore.sh \
  backups/<approved-backup>.dump \
  --confirm-disposable-restore
```

Require `DISPOSABLE_RESTORE_PASS` and expected Alembic head `20260801_001`.

## Browser canary

On desktop and a 390-pixel mobile viewport, verify in English and Persian RTL:

- WordPress queued, retrying, failed, reconciled, and completed states;
- Search Console disconnected, connected, property-selection, syncing, quota,
  revoked, failed, and completed states;
- no horizontal overflow;
- usable keyboard focus and accessible labels;
- no uncaught console errors or failed asset requests;
- safe user-facing messages without credentials or internal stack traces.

## Promotion decision

Promote only when every canary item passes, no duplicate WordPress post exists,
no unauthorized live publication is possible, the backup restore succeeds, and
no unresolved severity-1 or severity-2 issue remains.
