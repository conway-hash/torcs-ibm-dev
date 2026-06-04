White Monster Racing Team custom livery for car.
Use in case of mass start in evaluation period of the competition.
For video upload to competition use IBM livery ONLY!

In order to modify the livery of the car driven by `scr-server` either:

- In `torcs/drivers/scr_server/0/` overwrite `car1-ow1.rgb` with the file included.
- In order to keep original IBM livery upload the included `car1-ow1.rgb` under other name and modify the `torcs/drivers/scr_server/0/scr_server.xml` accordinly:

```
<attstr name="car name" val="car1-ow1"></attstr> <-- Modify val to the new .rgb name -->
```